# ML Research Sandbox for agent_framework
#
# Builds a container with the full ML stack, GPU support, and the sandbox server.
# All packages are installed at build time (network isolation at runtime).
#
# Build (from repo root):
#   docker build -f agent_framework/research.Dockerfile -t research-sandbox:latest .
#
# Run (mount data + GPU 0 only):
#   docker run -d --name research-sandbox \
#     --gpus '"device=0"' \
#     --network none \
#     --cap-drop ALL \
#     --security-opt no-new-privileges \
#     --read-only \
#     --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g \
#     -v /path/to/market/data:/data:ro \
#     -v /path/to/repo/0:/repo:ro \
#     -v /tmp/agent_workspace:/workspace:rw \
#     --user $(id -u):$(id -g) \
#     research-sandbox:latest
#
# Contains:
# - Python 3.12 with full ML stack (PyTorch 2.9+cu128, TF 2.21, Keras 3.15)
# - CUDA 12.8 toolkit for GPU access on GPU 0
# - model_flow codebase (sandbox server)
# - agent_framework codebase (research_sandbox adapter)
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /research

# Copy and install ML dependencies
COPY agent_framework/requirements_ml.txt /tmp/requirements_ml.txt
RUN pip install --no-cache-dir -r /tmp/requirements_ml.txt

# Copy project code
COPY model_flow/ /research/model_flow/
COPY agent_framework/ /research/agent_framework/

# Default command: start the sandbox server
CMD ["python3", "-m", "model_flow.sandbox.server"]