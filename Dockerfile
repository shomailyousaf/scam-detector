# Reproducible environment: same Python + same pinned libraries everywhere.
# The repository is mounted at /app at run time (see `make docker-reproduce`),
# so outputs are written back to your own folder.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the mounted repo is owned by the host user; let git inside the container read it
RUN git config --global --add safe.directory /app

CMD ["make", "reproduce"]
