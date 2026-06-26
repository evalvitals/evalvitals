FROM python:3.11-slim

WORKDIR /app

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[api]" Pillow

WORKDIR /app/example
COPY examples/analyzer_demos/hallucination/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
