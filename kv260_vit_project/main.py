import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from camera import Camera
from model import SimpleViTEnhancer
import time
import os

def main():
    # Configuration
    # KV260 often has limited memory, so keep resolution modest for ViT
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 480
    MODEL_INPUT_SIZE = (256, 256) # Resize for model input
    DEVICE = "cpu" # KV260 CPU.

    # Initialize Camera
    # Use index 0 for USB webcam, or a GStreamer string for MIPI
    # Example for MIPI:
    # temp_cam = Camera()
    # pipeline = temp_cam.get_gstreamer_pipeline()
    # cam = Camera(source=pipeline, width=1920, height=1080)

    print("Initializing camera...")
    cam = Camera(source=0, width=INPUT_WIDTH, height=INPUT_HEIGHT)

    if not cam.open():
        print("Error: Could not open camera. Please check connection.")
        # Create a dummy image for testing if camera fails, so the pipeline can be verified
        print("Creating dummy image for testing pipeline...")
        frame = np.random.randint(0, 255, (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    else:
        print("Capturing image...")
        # Warmup camera
        for _ in range(5):
            cam.read()

        frame = cam.read()
        if frame is None:
            print("Error: Could not read frame from camera.")
            cam.release()
            return
        cam.release()

    # Save original
    cv2.imwrite("original.jpg", frame)
    print("Saved original.jpg")

    # Initialize Model
    print("Loading Vision Transformer model...")
    model = SimpleViTEnhancer()
    model.to(DEVICE)
    model.eval()

    # Transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
    ])

    # Preprocess
    # OpenCV is BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(DEVICE)

    # Inference
    print("Running inference (Image Enhancement)...")
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.4f} seconds")

    # Postprocess
    output_tensor = output_tensor.squeeze(0).cpu()
    output_image = transforms.ToPILImage()(output_tensor)

    # Save enhanced
    output_image.save("enhanced.jpg")
    print("Saved enhanced.jpg")
    print("Project run complete.")

if __name__ == "__main__":
    main()
