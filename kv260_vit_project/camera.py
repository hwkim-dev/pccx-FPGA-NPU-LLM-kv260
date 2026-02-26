import cv2

class Camera:
    def __init__(self, source=0, width=640, height=480, framerate=30):
        self.source = source
        self.width = width
        self.height = height
        self.framerate = framerate
        self.cap = None

    def get_gstreamer_pipeline(self, device="/dev/video0"):
        # Example pipeline for Xilinx KV260 using v4l2src
        # Note: You might need to configure media-ctl before running this for MIPI cameras.
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw, width={self.width}, height={self.height}, framerate={self.framerate}/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )

    def open(self):
        # Try opening
        if isinstance(self.source, int):
             self.cap = cv2.VideoCapture(self.source)
             # Attempt to set resolution
             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif isinstance(self.source, str):
             self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)

        if not self.cap or not self.cap.isOpened():
             print(f"Failed to open camera source {self.source}")
             return False

        return True

    def read(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    # Test camera
    cam = Camera(source=0) # Change to string for GStreamer if needed
    if cam.open():
        print("Camera opened successfully")
        frame = cam.read()
        if frame is not None:
            print(f"Captured frame of shape: {frame.shape}")
            cv2.imwrite("test_capture.jpg", frame)
            print("Saved test_capture.jpg")
        cam.release()
    else:
        print("Could not open camera")
