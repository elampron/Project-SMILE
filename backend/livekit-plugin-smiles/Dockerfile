# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the SMILES plugin
COPY . /app/

# Install the plugin and its dependencies
RUN pip install -e .

# Install PyTorch (CPU version to keep image size smaller)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Download required files first
RUN python examples/smiles_assistant.py download-files

# Run the assistant in dev mode
CMD ["python", "examples/smiles_assistant.py", "dev"]
