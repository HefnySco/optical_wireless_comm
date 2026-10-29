import argparse
import time
import cv2
import numpy as np


def main():
  parser = argparse.ArgumentParser(
      description='Draw and pulse a colored rectangle on the monitor.'
  )
  parser.add_argument(
      '--rate',
      type=float,
      default=2.0,
      help='Pulse frequency in Hz (e.g., 2.0 for 2 flashes per second)',
  )
  parser.add_argument('--r', type=int, default=255, help='Red value (0-255)')
  parser.add_argument('--g', type=int, default=255, help='Green value (0-255)')
  parser.add_argument('--b', type=int, default=255, help='Blue value (0-255)')
  parser.add_argument(
      '--width', type=int, default=400, help='Width of the rectangle in pixels'
  )
  parser.add_argument(
      '--height',
      type=int,
      default=400,
      help='Height of the rectangle in pixels',
  )
  args = parser.parse_args()

  window_name = 'Pulsing Transmitter'
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(
      window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
  )

  # Full HD canvas (adjust if you have a 4K monitor)
  screen_w, screen_h = 1920, 1080
  canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

  base_color = np.array(
      [args.b, args.g, args.r], dtype=float
  )  # OpenCV uses BGR
  start_time = time.time()

  print(
      f'[*] Pulsing at {args.rate} Hz with RGB({args.r}, {args.g}, {args.b}).'
      " Press 'q' to quit."
  )

  while True:
    elapsed = time.time() - start_time

    # Sine wave modulation smoothly fades brightness between 0.0 and 1.0
    intensity = 0.5 * (1.0 + np.sin(2 * np.pi * args.rate * elapsed))
    current_color = base_color * intensity

    # Clear screen to black
    canvas[:] = 0

    # Calculate center coordinates for the rectangle
    x1 = (screen_w - args.width) // 2
    y1 = (screen_h - args.height) // 2
    x2 = x1 + args.width
    y2 = y1 + args.height

    # Draw filled rectangle
    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        tuple(current_color.astype(int).tolist()),
        -1,
    )

    cv2.imshow(window_name, canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()