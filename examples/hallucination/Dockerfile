FROM python:3.11-slim

WORKDIR /app

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals[api]"

WORKDIR /app/example
COPY examples/hallucination/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
