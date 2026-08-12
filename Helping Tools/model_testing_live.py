import argparse
from collections import Counter, deque
import json
import time
import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
ONNX_PATH = "black_connector.onnx"
CLASSES_PATH = "black_connector_classes.json"
CAMERA_INDEX = 0  # 0 or 1
SMOOTHING_WINDOW = 8
RESIZE_WIDTH = 640


# =========================================
# ONNX / OpenCV Helper Functions
# =========================================
def load_onnx_model(onnx_path, classes_path):
  net = cv2.dnn.readNetFromONNX(onnx_path)

  with open(classes_path, "r") as f:
    class_names = json.load(f)

  return net, class_names


def preprocess_frame(frame_bgr):
  resized = cv2.resize(frame_bgr, (224, 224))
  rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
  img = rgb.astype(np.float32) / 255.0

  mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
  img = (img - mean) / std

  # HWC -> CHW -> Batch (1, 3, 224, 224)
  img = img.transpose((2, 0, 1))
  img = np.expand_dims(img, axis=0)
  return img


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=1, keepdims=True)


def predict_frame(frame_bgr, net, class_names):
  blob = preprocess_frame(frame_bgr)
  net.setInput(blob)
  logits = net.forward()

  probs = softmax(logits)
  predicted_idx = np.argmax(probs, axis=1)[0]
  confidence = probs[0][predicted_idx]

  return class_names[predicted_idx], float(confidence)


def draw_overlay(frame, label, confidence, fps, smoothed_label=None):
  h, w = frame.shape[:2]

  overlay = frame.copy()
  cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
  frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

  color = (
      (0, 255, 0)
      if confidence > 0.6
      else (0, 165, 255)
      if confidence > 0.35
      else (0, 0, 255)
  )

  cv2.putText(
      frame,
      f"{label}  ({confidence*100:.1f}%)",
      (15, 35),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      color,
      2,
  )

  if smoothed_label is not None:
    cv2.putText(
        frame,
        f"stable: {smoothed_label}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

  cv2.putText(
      frame,
      f"FPS: {fps:.1f}",
      (w - 130, 35),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 255, 255),
      2,
  )

  return frame


# =========================================
# Main Loop
# =========================================
def main(onnx_path, classes_path, camera_index, smoothing_window, resize_width):
  net, class_names = load_onnx_model(onnx_path, classes_path)
  print(f"Loaded ONNX model with classes: {class_names}")

  cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

  if not cap.isOpened():
    cap = cv2.VideoCapture(camera_index)

  if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera at index {camera_index}. Check connections."
    )

  recent_predictions = deque(maxlen=smoothing_window)
  prev_time = time.time()
  snapshot_count = 0

  print("Press 'q' to quit, 's' to save a snapshot.")

  while True:
    ret, frame = cap.read()
    if not ret or frame is None:
      print("WARNING: failed to read frame from camera, stopping.")
      break

    if frame.shape[1] > resize_width:
      scale = resize_width / frame.shape[1]
      frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))

    label, confidence = predict_frame(frame, net, class_names)
    recent_predictions.append(label)
    smoothed_label = Counter(recent_predictions).most_common(1)[0][0]

    now = time.time()
    fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
    prev_time = now

    display_frame = draw_overlay(frame, label, confidence, fps, smoothed_label)
    cv2.imshow("Connector Classifier - Live", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    elif key == ord("s"):
      snapshot_count += 1
      out_path = f"snapshot_{snapshot_count}_{label}.jpg"
      cv2.imwrite(out_path, frame)
      print(f"Saved {out_path}")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--onnx", default=ONNX_PATH, help="Path to .onnx model")
  parser.add_argument(
      "--classes", default=CLASSES_PATH, help="Path to classes.json"
  )
  parser.add_argument(
      "--camera", type=int, default=CAMERA_INDEX, help="Camera index"
  )
  parser.add_argument(
      "--smoothing_window",
      type=int,
      default=SMOOTHING_WINDOW,
      help="Smoothing window size",
  )
  parser.add_argument(
      "--resize_width",
      type=int,
      default=RESIZE_WIDTH,
      help="Display resize width",
  )
  args = parser.parse_args()

  main(
      args.onnx,
      args.classes,
      args.camera,
      args.smoothing_window,
      args.resize_width,
  )
