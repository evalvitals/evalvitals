FROM python:3.11-slim

WORKDIR /app

COPY . /evalvitals
RUN pip install --no-cache-dir "/evalvitals"

WORKDIR /app/example
COPY examples/stats_compare/ .

ENV PYTHONUNBUFFERED=1

CMD ["python", "run.py"]
