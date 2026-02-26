import torch
import numpy as np
import cv2
import os
import sys

# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SimpleViTEnhancer
import torchvision.transforms as transforms

def test_pipeline():
    print("Testing pipeline...")

    # 1. Create dummy image (256x256 RGB)
    img_size = 256
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    cv2.imwrite("test_input.jpg", dummy_img)
    print("Created test_input.jpg")

    # 2. Load Model
    try:
        model = SimpleViTEnhancer()
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Preprocess
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), # (C, H, W) [0, 1]
    ])
    input_tensor = preprocess(dummy_img).unsqueeze(0) # (1, C, H, W)
    print(f"Input tensor shape: {input_tensor.shape}")

    # 4. Inference
    try:
        with torch.no_grad():
            output_tensor = model(input_tensor)
        print(f"Output tensor shape: {output_tensor.shape}")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # Verify shape before squeezing
    if output_tensor.shape == input_tensor.shape:
        print("Shape verification passed.")
    else:
        print(f"Shape mismatch: {output_tensor.shape} vs {input_tensor.shape}")

    # 5. Postprocess
    output_tensor_squeezed = output_tensor.squeeze(0)
    output_img = transforms.ToPILImage()(output_tensor_squeezed)
    output_img.save("test_output.jpg")
    print("Saved test_output.jpg")

    print("Test passed successfully.")

if __name__ == "__main__":
    test_pipeline()
