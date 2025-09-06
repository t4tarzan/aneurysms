FROM python:3.9-slim

WORKDIR /app

# Set pip to use a more reliable mirror and increase timeout
RUN pip config set global.timeout 1000 && \
    pip config set global.retries 10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY hf_requirements.txt .

# Install requirements in batches to avoid timeouts
RUN pip install --no-cache-dir numpy pillow pydicom && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r hf_requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /app/images

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "hf_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
