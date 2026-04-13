import cv2
import time

STREAM_URL = "http://192.168.1.144:81/stream"

def preprocess_frame(frame, target_size=(32, 32)):
    resized = cv2.resize(frame, target_size)
    normalized = resized.astype("float32") / 255.0
    return normalized

def main():
    print("Opening stream with OpenCV...")

    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Failed to open stream")
        return

    frame_count = 0
    start = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1

            processed = preprocess_frame(frame)

            if frame_count % 20 == 0:
                elapsed = time.time() - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                h, w = frame.shape[:2]
                print(f"Frame {frame_count}: {w}x{h}, FPS: {fps:.2f}")
                print(f"Processed shape: {processed.shape}")

            if frame_count == 50:
                cv2.imwrite("test_frame.jpg", frame)
                print("Saved test_frame.jpg")
                break

    finally:
        cap.release()

if __name__ == "__main__":
    main()