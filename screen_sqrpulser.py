import argparse
import time
import cv2
import numpy as np


def main():
  parser = argparse.ArgumentParser(
      description='Draw and pulse a square-wave rectangle on the monitor.'
  )
  parser.add_argument(
      '--rate',
      type=float,
      default=2.0,
      help='Pulse frequency in Hz (e.g., 2.0)',
  )
  parser.add_argument('--r', type=int, default=255, help='Red value (0-255)')
  parser.add_argument('--g', type=int, default=255, help='Green value (0-255)')
  parser.add_argument('--b', type=int, default=255, help='Blue value (0-255)')
  parser.add_argument(
      '--width', type=int, default=400, help='Width in pixels'
  )
  parser.add_argument(
      '--height', type=int, default=400, help='Height in pixels'
  )
  args = parser.parse_args()

  window_name = 'Square Wave Transmitter'
  cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(
      window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
  )

  screen_w, screen_h = 1920, 1080
  canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
  base_color = np.array([args.b, args.g, args.r], dtype=float)

  start_time = time.time()
  print(
      f'[*] Transmitting SQUARE wave at {args.rate} Hz. Press \'q\' to quit.'
  )

  while True:
    elapsed = time.time() - start_time

    # Square wave logic: 50% duty cycle (ON for half the period, OFF for half)
    period = 1.0 / args.rate
    phase = (elapsed % period) / period
    intensity = 1.0 if phase < 0.5 else 0.0

    current_color = base_color * intensity
    canvas[:] = 0

    x1 = (screen_w - args.width) // 2
    y1 = (screen_h - args.height) // 2
    x2 = x1 + args.width
    y2 = y1 + args.height

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