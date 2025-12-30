# mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: mlflow
        image: mlflow/mlflow:latest  # Or build custom
        args: ["mlflow", "server", "--host", "0.0.0.0", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/data/artifacts"]
        ports:
        - containerPort: 5000
        volumeMounts:
        - mountPath: /data
          name: mlflow-storage
      volumes:
      - name: mlflow-storage
        persistentVolumeClaim:
          claimName: mlflow-pvc  # On 8TB
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
spec:
  ports:
  - port: 5000
    targetPort: 5000
  selector:
    app: mlflow
