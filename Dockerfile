FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Minimal deps
RUN apt-get update && apt-get install -y python3-pip libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Torch with CUDA 12.1 wheels
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# YOLO + your deps
RUN pip3 install --no-cache-dir ultralytics paho-mqtt python-dotenv mlflow opencv-python

COPY app.py /app/app.py
WORKDIR /app
CMD ["python3", "-u", "app.py"]
