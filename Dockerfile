FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install torch with CUDA 12.1
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install deps
RUN pip3 install --no-cache-dir ultralytics paho-mqtt python-dotenv mlflow opencv-python

# Create Ultralytics cache dir
RUN mkdir -p /root/.cache/torch/hub

# Copy pre-downloaded models into the expected cache location
COPY models/yolo11n.pt /root/.cache/torch/hub/yolo11n.pt
COPY models/yolo11s.pt /root/.cache/torch/hub/yolo11s.pt
COPY models/yolo11m.pt /root/.cache/torch/hub/yolo11m.pt
# Add more if you downloaded them

# Optional: settings dir to silence warning
RUN mkdir -p /root/.config/Ultralytics

COPY src/app.py /app/app.py
WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python3", "-u", "app.py"]
