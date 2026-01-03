# detector2
##detector project to run with a Frigate network and Nvidia acceleration
(cloned from detection1)

This is a container for running ultralytics yolo containers on a frigate setup.  The idea is that we can use variations on a particular yolo model, focusing on yolo11 to start with

It runs cleanly in the my homelab setup (mini pc, lots of ram and disk, 1080 Ti GPU).  The following features work:
- written in python, all mainline logic is in src/app.py
- connects to MQTT, responds to all frigate topics
- a slice of the GPU is allocated
- has access to a 500GB PVC
- runs a yolo model, generates json and image artifacts 
- saves resulting image, logs it to MLFlow

