FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY service.py client.py /app/
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn","service:app","--host","0.0.0.0","--port","8080","--timeout-graceful-shutdown","45","--no-access-log"]
