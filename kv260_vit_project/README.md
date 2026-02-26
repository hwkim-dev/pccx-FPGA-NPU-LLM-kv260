# KV260 Vision Transformer Camera Enhancement Project

This project implements a simple image enhancement pipeline using a Vision Transformer (ViT) on the Xilinx Kria KV260 Vision AI Starter Kit. It captures images from a connected camera, processes them through a lightweight CNN-ViT hybrid model to enhance quality (e.g., denoising, super-resolution), and saves the result.

## Prerequisites

- **Hardware**: Xilinx Kria KV260 Vision AI Starter Kit.
- **OS**: Ubuntu 20.04/22.04 LTS or Petalinux for KV260.
- **Camera**: USB Webcam (default) or MIPI CSI-2 Camera (requires GStreamer configuration).

## Installation

1.  Clone this repository or copy the `kv260_vit_project` directory to your board.
2.  Install the required Python packages:

    ```bash
    pip3 install -r requirements.txt
    ```

    Note: On some embedded systems, you might need to install system dependencies for OpenCV and PyTorch (e.g., `libopencv-dev`).

## Usage

Run the main script:

```bash
python3 main.py
```

The script will:
1.  Attempt to open the default camera (index 0).
2.  Capture a single frame.
3.  Save the original frame as `original.jpg`.
4.  Process the frame using the Vision Transformer model.
5.  Save the enhanced frame as `enhanced.jpg`.

## Configuration

- **Camera Source**: By default, `main.py` uses `cv2.VideoCapture(0)`. If you are using a MIPI camera (like the IAS module), you may need to use a GStreamer pipeline.
    - Edit `main.py` to use `camera.get_gstreamer_pipeline()` if needed.
    - Ensure `media-ctl` is configured correctly for your sensor before running.

- **Model**: The `SimpleViTEnhancer` in `model.py` is a lightweight demo model. For real-world usage, you would train this model on a dataset of low/high-quality image pairs and load the weights.

## Troubleshooting

- **"Could not open camera"**: Check if your camera is connected and recognized (`ls /dev/video*`). If using MIPI, ensure the platform drivers are loaded (`xmutil listapps`).
- **Slow Inference**: The model runs on CPU by default. For real-time performance, the model should be quantized and compiled for the DPU (Deep Learning Processor Unit) using Vitis AI, which is beyond the scope of this Python-only demo.
