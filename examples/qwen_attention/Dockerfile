FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install evalvitals with local (white-box) dependencies
COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[local]"

WORKDIR /app/example
COPY examples/qwen_attention/ .

# Model weights are expected to be mounted at /models or pulled via HF_HOME.
# Set HF_HOME to a mounted volume to avoid re-downloading on every run.
ENV HF_HOME=/models
ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]