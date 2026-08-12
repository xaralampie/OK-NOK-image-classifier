import os
import time
import cv2


def get_camera():
  for index in [0, 2]:  # Checks index 1 first (common for external USB)
    # cv2.CAP_DSHOW speeds up initialization on Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
      # Test if we can actually read a frame
      ret, frame = cap.read()
      if ret:
        print(f'Successfully connected to camera at index {index}.')
        return cap
      cap.release()
  return None


def main():
  output_dir = 'dataa'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  cap = get_camera()

  if cap is None:
    print('Error: Could not find or open Logitech USB camera.')
    print('Check USB cable connection and camera privacy settings.')
    return

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  print('\n--- Logitech Feed Active ---')
  print('Press [SPACEBAR] to capture an image.')
  print('Press [Q] or [ESC] to quit.')

  img_counter = 0

  while True:
    ret, frame = cap.read()
    if not ret:
      print('Error: Failed to grab frame from Logitech camera.')
      break

    cv2.imshow('Logitech USB Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    # Spacebar key
    if key == 32:
      timestamp = time.strftime('%Y%m%d_%H%M%S')
      img_name = f'capture_{timestamp}_{img_counter:03d}.png'
      img_path = os.path.join(output_dir, img_name)

      cv2.imwrite(img_path, frame)
      print(f'[SAVED] {img_path}')
      img_counter += 1

    elif key in (ord('q'), 27):
      print('Exiting feed...')
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
