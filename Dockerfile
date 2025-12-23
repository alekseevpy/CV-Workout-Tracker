FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /backend

# Install root requirements
COPY backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy only specified folders (models accessible via relative paths)
COPY backend/ ./backend/

# Expose Streamlit
EXPOSE 8501
WORKDIR /backend/backend
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
