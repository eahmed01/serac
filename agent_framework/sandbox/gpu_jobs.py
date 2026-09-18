"""
GPU job management — multiprocessing GPU job spawning with resource tracking.

Pins child processes to specific GPUs via CUDA_VISIBLE_DEVICES and caps
VRAM usage via TensorFlow logical device configuration.

Uses multiprocessing start_method='spawn' for CUDA safety (fork is not
safe with CUDA contexts). Provides resource coordination only — does NOT
provide security isolation.
"""

import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class GPUResource:
    """GPU resource configuration for child processes.

    Parameters
    ----------
    device : int, list of int, or str
        GPU device ID(s). E.g. 0, [0, 1], or "0,1,2" for multi-GPU.
    max_memory_gb : float
        Maximum VRAM in GB. 0 = unlimited.
    extra_env : dict
        Additional environment variables to set for the child process.
    """

    device: Union[int, List[int], str] = 0
    max_memory_gb: float = 4.0
    extra_env: Dict[str, str] = field(default_factory=dict)

    @property
    def cuda_visible_devices(self) -> str:
        if isinstance(self.device, int):
            return str(self.device)
        if isinstance(self.device, str):
            return self.device
        return ",".join(str(d) for d in self.device)


@dataclass
class GPUJob:
    """A spawned GPU job."""

    job_id: str
    name: str
    script: str
    status: str = "running"  # running, completed, failed, killed
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    pid: Optional[int] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()


def _run_gpu_job(
    gpu_resource: GPUResource,
    script: str,
    job_id: str,
    output_dir: str,
) -> None:
    """Worker function: run a Python script with GPU environment set.

    This runs in a child process with GPU isolation.
    """
    # Ensure any further multiprocessing from within this process also uses spawn
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)

    # Set GPU environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_resource.cuda_visible_devices
    if gpu_resource.max_memory_gb > 0:
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Signal to child subprocesses that the parent used spawn method
    env["GPUJOBS_START_METHOD"] = "spawn"

    # Apply extra env
    env.update(gpu_resource.extra_env)

    # Capture output
    stdout_path = os.path.join(output_dir, f"{job_id}.stdout.log")
    stderr_path = os.path.join(output_dir, f"{job_id}.stderr.log")

    # If max_memory_gb > 0, generate a wrapper script that enforces VRAM cap
    # via TensorFlow logical device configuration
    wrapper_script = None
    if gpu_resource.max_memory_gb > 0:
        memory_limit_mb = int(gpu_resource.max_memory_gb * 1024)
        wrapper_content = (
            f"import sys\n"
            f"# VRAM cap: {gpu_resource.max_memory_gb} GB = {memory_limit_mb} MB\n"
            f"try:\n"
            f"    import tensorflow as tf\n"
            f"    gpus = tf.config.list_physical_devices('GPU')\n"
            f"    if gpus:\n"
            f"        tf.config.set_logical_device_configuration(\n"
            f"            gpus[0],\n"
            f"            [tf.config.LogicalDeviceConfiguration(memory_limit={memory_limit_mb})]\n"
            f"        )\n"
            f"except ImportError:\n"
            f"    pass  # TensorFlow not available; VRAM cap not enforced\n"
            f"except Exception:\n"
            f"    pass  # TF init failed; continue without cap\n"
            f"# Execute the user script\n"
            f"exec(compile(open({script!r}).read(), {script!r}, 'exec'))\n"
        )
        # Write wrapper to a temp file in the output dir
        wrapper_path = os.path.join(output_dir, f"{job_id}.wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(wrapper_content)
        wrapper_script = wrapper_path

    effective_script = wrapper_script or script

    returncode = -1
    try:
        with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
            result = subprocess.run(
                [sys.executable, effective_script],
                env=env,
                stdout=stdout,
                stderr=stderr,
                cwd=os.path.dirname(os.path.abspath(script)),
            )
            returncode = result.returncode

            # Write status
            status_path = os.path.join(output_dir, f"{job_id}.status")
            with open(status_path, "w") as f:
                f.write(f"{returncode}\n")
    except Exception as e:
        status_path = os.path.join(output_dir, f"{job_id}.status")
        with open(status_path, "w") as f:
            f.write(f"-1\n{e}\n")

    # Propagate the subprocess exit code so the monitor thread sees the
    # correct status (non-zero exit → "failed", not "completed")
    if returncode != 0:
        sys.exit(returncode)


_MAX_RETAINED_JOBS = 100


class GPUManager:
    """Manage GPU job lifecycle.

    Parameters
    ----------
    gpu_resource : GPUResource
        GPU configuration for child processes.
    output_dir : str or None
        Directory for job output files. None = auto-generated temp dir.
    max_retained_jobs : int
        Maximum number of completed/failed/killed jobs to keep in memory.
        Oldest are evicted first. Default 100.
    """

    def __init__(
        self,
        gpu_resource: Optional[GPUResource] = None,
        output_dir: Optional[str] = None,
        max_retained_jobs: int = _MAX_RETAINED_JOBS,
    ) -> None:
        self.gpu_resource = gpu_resource or GPUResource()
        self._output_dir = output_dir or str(Path(tempfile.gettempdir()) / "sandbox_gpu_jobs")
        self.active_jobs: Dict[str, GPUJob] = {}
        self._processes: Dict[str, Any] = {}
        self._max_retained_jobs = max_retained_jobs
        self._lock = threading.Lock()

        # Ensure output dir exists
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

    def spawn_job(self, name: str, script: str) -> str:
        """Spawn a GPU job.

        Parameters
        ----------
        name : str
            Job name for tracking.
        script : str
            Path to Python script to execute.

        Returns
        -------
        str
            Job ID.

        Raises
        ------
        FileNotFoundError
            If the script path does not exist.
        """
        # Validate script exists before spawning
        if not os.path.isfile(script):
            raise FileNotFoundError(f"Script not found: {script}")

        job_id = f"{name}_{uuid.uuid4().hex[:8]}"
        job = GPUJob(
            job_id=job_id,
            name=name,
            script=script,
        )

        # Spawn process using 'spawn' start method (CUDA-safe)
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_run_gpu_job,
            args=(self.gpu_resource, script, job_id, self._output_dir),
        )
        proc.start()
        job.pid = proc.pid
        self.active_jobs[job_id] = job
        self._processes[job_id] = proc

        # Start background monitor thread
        monitor = threading.Thread(target=self._monitor_job, args=(job_id, proc), daemon=True)
        monitor.start()

        return job_id

    def _monitor_job(self, job_id: str, proc: multiprocessing.Process) -> None:
        """Monitor a job until completion (runs in background thread)."""
        while proc.is_alive():
            proc.join(timeout=0.1)

        # Update job status with lock to prevent race with kill_job()
        with self._lock:
            job = self.active_jobs.get(job_id)
            if job is None:
                return
            # If already killed by kill_job(), don't overwrite status
            if job.status == "killed":
                return
            if proc.exitcode == 0:
                job.status = "completed"
            else:
                job.status = "failed"

            job.exit_code = proc.exitcode
            job.end_time = datetime.now(timezone.utc)

            # Read output files
            stdout_path = os.path.join(self._output_dir, f"{job_id}.stdout.log")
            stderr_path = os.path.join(self._output_dir, f"{job_id}.stderr.log")
            if os.path.exists(stdout_path):
                job.stdout = Path(stdout_path).read_text()
            if os.path.exists(stderr_path):
                job.stderr = Path(stderr_path).read_text()

    def kill_job(self, job_id: str) -> bool:
        """Kill a running job.

        Returns True if the job was found and killed.
        """
        with self._lock:
            job = self.active_jobs.get(job_id)
            if job and job.status == "running":
                proc = self._processes.get(job_id)
                if proc and proc.is_alive():
                    proc.kill()
                    proc.join()  # Reap zombie process
                job.status = "killed"
                job.end_time = datetime.now(timezone.utc)
                return True
        return False

    def get_job(self, job_id: str) -> Optional[GPUJob]:
        """Get job details."""
        return self.active_jobs.get(job_id)

    def list_jobs(self, status: Optional[str] = None) -> List[GPUJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.active_jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.start_time, reverse=True)

    def cleanup(self, max_age_hours: Optional[float] = None) -> int:
        """Remove old completed/failed/killed jobs.

        Parameters
        ----------
        max_age_hours : float or None
            If set, remove finished jobs older than this many hours.
            If None, only enforce max_retained_jobs limit.

        Returns
        -------
        int
            Number of jobs removed.
        """
        removed = 0
        now = datetime.now(timezone.utc)

        with self._lock:
            finished_statuses = {"completed", "failed", "killed"}
            finished = {
                jid: j
                for jid, j in self.active_jobs.items()
                if j.status in finished_statuses
            }

            # Filter by age if requested
            if max_age_hours is not None:
                to_remove = {
                    jid: j
                    for jid, j in finished.items()
                    if j.end_time is not None
                    and (now - j.end_time).total_seconds() > max_age_hours * 3600
                }
            else:
                # Sort by end_time (oldest first) and trim excess
                sorted_finished = sorted(
                    finished.items(),
                    key=lambda item: (item[1].end_time or datetime.min.replace(tzinfo=timezone.utc)),
                )
                excess = len(sorted_finished) - self._max_retained_jobs
                if excess > 0:
                    to_remove = {jid: j for jid, j in sorted_finished[:excess]}
                else:
                    to_remove = {}

            for jid in to_remove:
                del self.active_jobs[jid]
                self._processes.pop(jid, None)
                removed += 1

        return removed

    @property
    def output_dir(self) -> str:
        return self._output_dir
