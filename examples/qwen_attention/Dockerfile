FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install evalvitals with local (white-box) dependencies.
# torch is already in the base image — install only the non-torch extras
# to avoid pip overwriting the CUDA wheel with a CPU one from PyPI.
COPY . /evalvitals
RUN pip install --no-cache-dir transformers>=4.51.0 accelerate && \
    pip install --no-cache-dir --no-deps "/evalvitals"

WORKDIR /app/example
COPY examples/qwen_attention/ .

# Model weights are expected to be mounted at /models or pulled via HF_HOME.
# Set HF_HOME to a mounted volume to avoid re-downloading on every run.
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]