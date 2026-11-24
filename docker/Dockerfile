# qwen38-plus image — builds and runs the C++/CUDA engine and its server on GB10 (sm_121a).
# Base (local on C, nothing pulled): NGC PyTorch 26.07 = CUDA 13.3 · cmake 3.31 · ninja · gcc 13 · NCCL · ibverbs/mlx5 ·
# nsys/ncu · torch/numpy/safetensors (reference checks against the golden dump) · CuTe DSL.
# Tag = qwen38-plus:<first 8 of this file's sha256>; a changed file is a new tag, old tags are never overwritten.
FROM nvcr.io/nvidia/pytorch:26.07-py3
LABEL project=qwen38-engine
ENV CUDAARCHS=121a \
    TORCH_CUDA_ARCH_LIST=12.1a \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# server front (bundle 4): HTTP only, pinned, no dependency resolution — tokenizers, jinja2, pydantic, anyio, click and
# h11 come with the base image
RUN pip install --no-deps fastapi==0.115.6 starlette==0.41.3 uvicorn==0.32.1 && \
    python -c "import fastapi, starlette, uvicorn, tokenizers, jinja2"
# vision front (bundle 3): image and video decoding pinned to the golden's versions (reference image: Pillow 13.0.0,
# opencv-python-headless 5.0.0.93 — its own FFmpeg) so the same file gives the same pixels; torch / torchvision (resize)
# come with the base image at the golden's versions (2.13 / 0.28)
RUN pip install --no-deps pillow==13.0.0 opencv-python-headless==5.0.0.93 && \
    python -c "import PIL, cv2; assert PIL.__version__ == '13.0.0' and cv2.__version__.startswith('5.0.0'), (PIL.__version__, cv2.__version__)"
# no NGC banner in docker logs: plain entrypoint
ENTRYPOINT []
WORKDIR /work/src
CMD ["bash"]
