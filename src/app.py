import os
import time
import mlflow
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import base64 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
import datetime
from ultralytics import YOLO

# Force flush just in case
sys.stdout.flush()
print(f"[{datetime.datetime.now()}] >>> DETECTION2 CONTAINER STARTED <<<")
print(f"[{datetime.datetime.now()}] Python version: {sys.version}")
print(f"[{datetime.datetime.now()}] Attempting MQTT connection to mosquitto.mqtt.svc.cluster.local:1883...")
sys.stdout.flush()

load_dotenv()  # For local dev; in k8s use Secrets

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service.mlflow.svc.cluster.local:5000"))
mlflow.set_experiment("detection-experiments")

# MQTT credentials (from Secrets in k8s)
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker.default.svc.cluster.local")  # adjust to your broker service
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = "frigate/#"

model = YOLO("/root/.cache/torch/hub/yolo11s.pt")

def check_gpu():
    print(f"[{datetime.datetime.now()}] GPU TEST START")
    if torch.cuda.is_available():
        print(f"CUDA available! Device: {torch.cuda.get_device_name(0)}")
        # Simple GPU op: matrix multiply
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        print(f"GPU matmul result sample: {c[0][0].item():.2f}")
        torch.cuda.synchronize()  # Ensure completion
    else:
        print("No CUDA – falling back to CPU")
    print(f"[{datetime.datetime.now()}] GPU TEST END")
    sys.stdout.flush()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
    else:
        print(f"Connection failed with code {rc}")
    check_gpu()

def on_message(client, userdata, msg):
    print(f"[{datetime.datetime.now()}] >>> RAW MQTT MESSAGE RECEIVED for Topic {msg.topic}<<<")
    
    if not msg.topic.endswith('snapshot'):
        return

    with mlflow.start_run(run_name="detection2-yolo"):
        mlflow.log_param("topic", msg.topic)
        image_bytes = msg.payload
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Flags for color image
        if img is None:
            print("Failed to decode image – corrupt payload?")
            return
        results = model(img)  # Runs on GPU automatically
        for r in results:
            print(f"Detected: {r.names[int(r.boxes.cls[0])]} at {r.boxes.xyxy[0].tolist()}")
        mlflow.log_param("detections", json.dumps(results))
        mlflow.log_metric("inference_time", inference_time)
        mlflow.log_artifact(artifact_path)

# Create and configure the client
client = mqtt.Client(client_id="detection2")
if MQTT_USER and MQTT_PASSWORD:
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# Connect and loop
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()  # Blocks here; handles reconnects automatically
