"""
GPU job management tests — async GPU job spawning with resource tracking.
"""

import multiprocessing
import os
import tempfile
import threading
import time
import pytest
from pathlib import Path
from agent_framework.sandbox.gpu_jobs import GPUManager, GPUResource


def _wait_for_status(mgr: GPUManager, job_id: str, expected_status: str, timeout: float = 10.0) -> None:
    """Poll until job reaches expected_status or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = mgr.get_job(job_id)
        if job and job.status == expected_status:
            return
        time.sleep(0.2)
    job = mgr.get_job(job_id)
    actual = job.status if job else "missing"
    raise AssertionError(
        f"Job {job_id} status is '{actual}', expected '{expected_status}' "
        f"after {timeout}s timeout"
    )


# Most tests use max_memory_gb=0 to skip the TensorFlow wrapper (which
# triggers expensive GPU PTX compilation on first import). Tests that
# specifically verify the wrapper use max_memory_gb > 0.
_FAST_RESOURCE = GPUResource(device=0, max_memory_gb=0)


class TestGPUResource:

    def test_default_resource(self):
        """Default GPUResource should have sensible defaults."""
        res = GPUResource()
        assert res.device == 0
        assert res.max_memory_gb == 4.0

    def test_custom_resource(self):
        """Custom GPUResource should preserve settings."""
        res = GPUResource(device=1, max_memory_gb=8.0)
        assert res.device == 1
        assert res.max_memory_gb == 8.0

    def test_cuda_visible_devices(self):
        """cuda_visible_devices should return correct string."""
        res = GPUResource(device=0)
        assert res.cuda_visible_devices == "0"

    def test_extra_env(self):
        """extra_env should be mergeable into process environment."""
        res = GPUResource(device=0, extra_env={"PYTHONPATH": "/custom"})
        assert "PYTHONPATH" in res.extra_env

    # Bug 7 fixes: multi-GPU device support
    def test_cuda_visible_devices_multi_gpu_list(self):
        """cuda_visible_devices should handle list of GPU IDs."""
        res = GPUResource(device=[0, 1, 2])
        assert res.cuda_visible_devices == "0,1,2"

    def test_cuda_visible_devices_multi_gpu_string(self):
        """cuda_visible_devices should pass through string device spec."""
        res = GPUResource(device="0,1,2")
        assert res.cuda_visible_devices == "0,1,2"

    def test_cuda_visible_devices_single_gpu_string(self):
        """cuda_visible_devices should handle single-GPU string."""
        res = GPUResource(device="3")
        assert res.cuda_visible_devices == "3"


class TestGPUManager:

    def test_create_manager(self):
        """GPUManager should initialize with default resource."""
        mgr = GPUManager()
        assert len(mgr.active_jobs) == 0

    def test_create_with_custom_resource(self):
        """GPUManager should accept custom GPUResource."""
        res = GPUResource(device=0, max_memory_gb=2.0)
        mgr = GPUManager(gpu_resource=res)
        assert mgr.gpu_resource.max_memory_gb == 2.0

    def test_job_lifecycle(self, tmp_path):
        """Job should spawn, complete, and be tracked."""
        script = tmp_path / "test_job.py"
        script.write_text("print('gpu_job_done')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("test_job", str(script))
        assert job_id is not None

        _wait_for_status(mgr, job_id, "completed")

        # Job should be tracked
        assert job_id in mgr.active_jobs

    def test_kill_job(self, tmp_path):
        """Killing a job should terminate the process."""
        script = tmp_path / "sleep_job.py"
        script.write_text("import time; time.sleep(60)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("sleep_job", str(script))

        # Give it time to start
        time.sleep(0.5)

        mgr.kill_job(job_id)
        assert job_id in mgr.active_jobs  # still tracked, but killed

    def test_concurrent_jobs(self, tmp_path):
        """Multiple jobs should be tracked independently."""
        script = tmp_path / "simple.py"
        script.write_text("print('ok')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        jobs = []
        for i in range(3):
            job_id = mgr.spawn_job(f"job_{i}", str(script))
            jobs.append(job_id)

        # All jobs should be tracked
        assert len(mgr.active_jobs) == 3
        for job_id in jobs:
            assert job_id in mgr.active_jobs

    def test_gpu_env_isolation(self, tmp_path):
        """GPU environment should be set correctly for child processes."""
        script = tmp_path / "check_env.py"
        script.write_text("import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', 'none'))\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("env_check", str(script))

        _wait_for_status(mgr, job_id, "completed")

        job = mgr.active_jobs[job_id]
        # The job should have captured the GPU env
        assert job.stdout or job.stderr

    # Bug 1 fix: spawn method verification
    def test_spawn_method_is_spawn(self, tmp_path):
        """Processes should be spawned with start_method='spawn' for CUDA safety."""
        script = tmp_path / "check_start_method.py"
        script.write_text(
            "import os\n"
            "print(os.environ.get('GPUJOBS_START_METHOD', 'not_set'))\n"
        )

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("check_method", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert job is not None
        assert job.status == "completed"
        assert "spawn" in job.stdout

    # Bug 2 fix: VRAM cap enforcement (wrapper script generation)
    def test_vram_cap_wrapper_generated(self, tmp_path):
        """When max_memory_gb > 0, a wrapper script with TF VRAM cap should be generated."""
        script = tmp_path / "dummy.py"
        script.write_text("print('hello')\n")

        output_dir = str(tmp_path / "gpu_out")
        res = GPUResource(device=0, max_memory_gb=2.0)
        mgr = GPUManager(gpu_resource=res, output_dir=output_dir)

        job_id = mgr.spawn_job("vram_test", str(script))
        _wait_for_status(mgr, job_id, "completed")

        # Check that the wrapper script was created
        wrapper_path = os.path.join(output_dir, f"{job_id}.wrapper.py")
        assert os.path.exists(wrapper_path), "Wrapper script should be created when max_memory_gb > 0"

        # Check wrapper content includes TF logical device configuration
        wrapper_content = Path(wrapper_path).read_text()
        assert "set_logical_device_configuration" in wrapper_content
        assert "memory_limit=2048" in wrapper_content  # 2.0 GB * 1024

    def test_no_wrapper_when_unlimited(self, tmp_path):
        """When max_memory_gb is 0, no wrapper script should be generated."""
        script = tmp_path / "dummy.py"
        script.write_text("print('hello')\n")

        output_dir = str(tmp_path / "gpu_out")
        res = GPUResource(device=0, max_memory_gb=0)
        mgr = GPUManager(gpu_resource=res, output_dir=output_dir)

        job_id = mgr.spawn_job("no_vram_test", str(script))
        _wait_for_status(mgr, job_id, "completed")

        wrapper_path = os.path.join(output_dir, f"{job_id}.wrapper.py")
        assert not os.path.exists(wrapper_path), "No wrapper when max_memory_gb=0"

    # Bug 3 fix: kill_job race condition with monitor
    def test_kill_job_status_not_overwritten(self, tmp_path):
        """kill_job status should not be overwritten by monitor thread."""
        script = tmp_path / "sleep_job.py"
        script.write_text("import time; time.sleep(60)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("long_sleep", str(script))

        # Give it time to start
        time.sleep(0.5)

        assert mgr.get_job(job_id).status == "running"

        # Kill the job
        killed = mgr.kill_job(job_id)
        assert killed is True

        # Wait for monitor to process the dead process
        time.sleep(1.0)

        job = mgr.get_job(job_id)
        assert job is not None
        # Status should remain "killed", not overwritten to "failed"
        assert job.status == "killed"

    # Bug 4 fix: zombie process cleanup (join after kill)
    def test_kill_job_reaps_process(self, tmp_path):
        """kill_job should call proc.join() to reap the process."""
        script = tmp_path / "sleep_job.py"
        script.write_text("import time; time.sleep(60)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("reap_test", str(script))
        time.sleep(0.5)

        mgr.kill_job(job_id)
        time.sleep(0.5)

        # The process should be reaped — check via the process dict
        proc = mgr._processes.get(job_id)
        if proc is not None:
            assert not proc.is_alive(), "Process should be dead after kill + join"

    # Bug 5 fix: unbounded job accumulation — cleanup
    def test_cleanup_evicts_old_jobs(self, tmp_path):
        """cleanup() should evict oldest finished jobs beyond max_retained_jobs."""
        script = tmp_path / "quick.py"
        script.write_text("print('done')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE, max_retained_jobs=3)

        # Spawn 5 jobs that complete quickly
        job_ids = []
        for i in range(5):
            jid = mgr.spawn_job(f"cleanup_{i}", str(script))
            job_ids.append(jid)

        # Wait for all to complete
        for jid in job_ids:
            _wait_for_status(mgr, jid, "completed")

        # All should be completed
        completed = mgr.list_jobs(status="completed")
        assert len(completed) == 5

        # Cleanup should evict oldest to keep only 3
        removed = mgr.cleanup()
        assert removed == 2

        remaining = mgr.list_jobs(status="completed")
        assert len(remaining) == 3

    def test_cleanup_by_age(self, tmp_path):
        """cleanup(max_age_hours) should remove jobs older than threshold."""
        script = tmp_path / "quick.py"
        script.write_text("print('done')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)

        job_id = mgr.spawn_job("age_test", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert job.status == "completed"

        # Manually set end_time far in the past
        from datetime import timedelta
        job.end_time = job.end_time - timedelta(hours=48)

        # Cleanup with 24-hour limit should remove it
        removed = mgr.cleanup(max_age_hours=24)
        assert removed == 1
        assert mgr.get_job(job_id) is None

    def test_cleanup_no_running_jobs_evicted(self, tmp_path):
        """cleanup() should never evict running jobs."""
        script = tmp_path / "sleep.py"
        script.write_text("import time; time.sleep(60)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE, max_retained_jobs=0)

        job_id = mgr.spawn_job("running", str(script))
        time.sleep(0.5)

        # cleanup should not evict running jobs
        removed = mgr.cleanup()
        assert removed == 0
        assert mgr.get_job(job_id) is not None

        # Clean up
        mgr.kill_job(job_id)

    # Bug 6 fix: script validation
    def test_spawn_job_raises_on_missing_script(self):
        """spawn_job should raise FileNotFoundError for non-existent scripts."""
        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        with pytest.raises(FileNotFoundError, match="Script not found"):
            mgr.spawn_job("bad_script", "/nonexistent/path/script.py")

    def test_spawn_job_raises_on_directory(self):
        """spawn_job should raise FileNotFoundError when script is a directory."""
        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        with pytest.raises(FileNotFoundError, match="Script not found"):
            mgr.spawn_job("is_dir", "/tmp")

    # Bug 7 fix: multi-GPU device strings
    def test_multi_gpu_device_propagated(self, tmp_path):
        """Multi-GPU device string should propagate to CUDA_VISIBLE_DEVICES."""
        script = tmp_path / "check_multi_gpu.py"
        script.write_text(
            "import os\n"
            "print(os.environ.get('CUDA_VISIBLE_DEVICES', 'none'))\n"
        )

        res = GPUResource(device="0,1,2", max_memory_gb=0)
        mgr = GPUManager(gpu_resource=res)

        job_id = mgr.spawn_job("multi_gpu", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert job is not None
        assert "0,1,2" in job.stdout

    # Additional missing tests from review
    def test_job_failure_detection(self, tmp_path):
        """Job with non-zero exit code should be marked as failed."""
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(42)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("fail_job", str(script))
        _wait_for_status(mgr, job_id, "failed")

        job = mgr.get_job(job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.exit_code == 42

    def test_job_stdout_stderr_content(self, tmp_path):
        """Job stdout/stderr should contain actual script output."""
        script = tmp_path / "echo.py"
        script.write_text(
            "print('hello stdout')\n"
            "import sys; sys.stderr.write('hello stderr')\n"
        )

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("echo", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert "hello stdout" in job.stdout
        assert "hello stderr" in job.stderr

    def test_kill_completed_job_returns_false(self, tmp_path):
        """kill_job on already-completed job should return False."""
        script = tmp_path / "quick.py"
        script.write_text("print('done')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("fast", str(script))
        _wait_for_status(mgr, job_id, "completed")

        assert mgr.kill_job(job_id) is False
        assert mgr.kill_job("nonexistent_id") is False

    def test_job_duration(self, tmp_path):
        """Job duration should be computed from start/end times."""
        script = tmp_path / "wait.py"
        script.write_text("import time; time.sleep(0.2)\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        job_id = mgr.spawn_job("wait", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert job.duration is not None
        assert 0.1 < job.duration < 10

    def test_output_files_created(self, tmp_path):
        """Job output files should be created in output_dir."""
        script = tmp_path / "test.py"
        script.write_text("print('test')\n")

        output_dir = str(tmp_path / "gpu_out")
        mgr = GPUManager(gpu_resource=_FAST_RESOURCE, output_dir=output_dir)

        job_id = mgr.spawn_job("files_test", str(script))
        _wait_for_status(mgr, job_id, "completed")

        assert os.path.exists(os.path.join(output_dir, f"{job_id}.stdout.log"))
        assert os.path.exists(os.path.join(output_dir, f"{job_id}.stderr.log"))
        assert os.path.exists(os.path.join(output_dir, f"{job_id}.status"))

    def test_tf_env_growth_set(self, tmp_path):
        """TF_FORCE_GPU_ALLOW_GROWTH should be set when max_memory_gb > 0."""
        script = tmp_path / "check_tf.py"
        script.write_text(
            "import os; print(os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'not_set'))\n"
        )

        res = GPUResource(device=0, max_memory_gb=4.0)
        mgr = GPUManager(gpu_resource=res)

        job_id = mgr.spawn_job("tf_check", str(script))
        _wait_for_status(mgr, job_id, "completed")

        job = mgr.get_job(job_id)
        assert "true" in job.stdout

    def test_list_jobs_filtering(self, tmp_path):
        """list_jobs(status=...) should filter correctly."""
        script = tmp_path / "quick.py"
        script.write_text("print('done')\n")

        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        for i in range(3):
            mgr.spawn_job(f"filtered_{i}", str(script))

        # Wait for all to complete
        for j in mgr.list_jobs():
            _wait_for_status(mgr, j.job_id, "completed")

        completed = mgr.list_jobs(status="completed")
        assert all(j.status == "completed" for j in completed)
        assert len(completed) == 3

    def test_thread_safety_lock_exists(self):
        """GPUManager should have a threading lock for job state mutations."""
        mgr = GPUManager(gpu_resource=_FAST_RESOURCE)
        assert hasattr(mgr, "_lock")
        assert isinstance(mgr._lock, type(threading.Lock()))
