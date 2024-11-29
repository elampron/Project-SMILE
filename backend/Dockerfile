# Use Python slim image with explicit platform support
FROM python:3.12-slim

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG=C.UTF-8

# runtime dependencies
RUN apt-get update && apt-get upgrade -y

WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt


EXPOSE 8002

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8002"]