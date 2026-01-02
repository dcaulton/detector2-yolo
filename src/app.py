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
mlflow.set_experiment("exp-2026-yolo-vit")

# MQTT credentials (from Secrets in k8s)
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker.default.svc.cluster.local")  # adjust to your broker service
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
MQTT_TOPIC = "frigate/#"

model = YOLO("/root/.cache/torch/hub/yolo11m.pt")

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

def extract_frigate_event_id(jpeg_bytes):
    from struct import unpack
    i = 2  # Skip SOI (FFD8 FF)
    while i < len(jpeg_bytes) - 1:
        marker = jpeg_bytes[i:i+2]
        if marker[0] != 0xFF:
            break
        i += 2
        length = unpack('>H', jpeg_bytes[i:i+2])[0]
        i += 2
        if marker[1] == 0xE1:  # APP1 (EXIF) or 0xEE (APP14)
            chunk = jpeg_bytes[i:i+length].decode('utf-8', errors='ignore')
            if 'frigate' in chunk.lower() or any(s in chunk for s in ['.', '-fx', '-ao']):  # Heuristic for ID pattern
                # Extract <timestamp>.<us>-<hex> (regex or split)
                import re
                mid = re.search(r'(\d+\.\d+-[a-z0-9]{6,8})', chunk)
                if mid:
                    return mid.group(1)
        i += length
    return None

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

    image_bytes = msg.payload
    event_id = extract_frigate_event_id(image_bytes) or f"{msg.topic}_{hash(msg.payload[:32])}"
    with mlflow.start_run(run_name=event_id):
        mlflow.log_param("topic", msg.topic)
        mlflow.log_param("event_id", event_id)
        mlflow.log_param("detector_type", "yolo")
        start_time = time.perf_counter()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Flags for color image
        if img is None:
            print("Failed to decode image – corrupt payload?")
            return
        results = model(img)  # Runs on GPU automatically
        end_time = time.perf_counter()
        inference_time = ((end_time - start_time) * 1000)
        mlflow.log_metric("inference_time", inference_time)

        r = results[0]  # The Results object for this image

        boxes = r.boxes  # Boxes tensor

        if len(boxes) == 0:
            print("No detections above confidence threshold")
            mlflow.log_metric("num_detections", 0)
        else:
            print(f"{len(boxes)} detection(s) found")
            mlflow.log_metric("num_detections", len(boxes))

            # Now safe to access top detection
            top_cls_id = int(boxes.cls[0].item())
            top_conf = boxes.conf[0].item()
            top_name = r.names[top_cls_id]
            top_bbox = boxes.xyxy[0].tolist()

            print(f"Top detection: {top_name} (confidence: {top_conf:.2f}) at {top_bbox}")

            # Optional: Log top detection as params/metrics
            mlflow.log_param("top_class", top_name)
            mlflow.log_metric("top_confidence", top_conf)

            # Annotated image (only if detections exist)
            annotated = r.plot()  # Draws boxes on image
            annotated_path = "/data/yolo_annotated.jpg"
            cv2.imwrite(annotated_path, annotated)

            try:
              mlflow.log_artifact(annotated_path)
              print("Artifact uploaded successfully")
            except Exception as e:
              print(f"MLflow artifact upload failed: {str(e)}")

            # Full detections JSON (as before)
            detections = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                detections.append({
                    "class_name": r.names[cls_id],
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": boxes.xyxy[i].tolist()
                })
            json_path = "/data/detections.json"
            with open(json_path, "w") as f:
                json.dump(detections, f, indent=2)
            mlflow.log_artifact(json_path)

# Create and configure the client
client = mqtt.Client(client_id="detection2")
if MQTT_USER and MQTT_PASSWORD:
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# Connect and loop
client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_forever()  # Blocks here; handles reconnects automatically
