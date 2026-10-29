import argparse
import time
from collections import deque
import cv2
import numpy as np


def main():
  parser = argparse.ArgumentParser(
      description='Detect square wave pulses and harmonics from a webcam stream.'
  )
  parser.add_argument(
      '--target-rate',
      type=float,
      default=2.0,
      help='Expected fundamental pulse frequency in Hz',
  )
  args = parser.parse_args()

  cap = cv2.VideoCapture('/dev/video0')
  if not cap.isOpened():
    print('Error: Could not open /dev/video0')
    return

  fps = cap.get(cv2.CAP_PROP_FPS)
  if fps <= 0:
    fps = 30.0
  print(f'[*] Webcam opened. Estimated FPS: {fps}')

  buffer_duration = 4.0
  max_len = int(fps * buffer_duration)
  intensity_history = deque(maxlen=max_len)
  time_history = deque(maxlen=max_len)

  print(
      "[*] Listening for square wave & harmonics... Press 'q' to quit."
  )

  while True:
    ret, frame = cap.read()
    if not ret:
      print('Error: Failed to grab frame.')
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    current_time = time.time()
    intensity_history.append(mean_intensity)
    time_history.append(current_time)

    top_peaks = []

    if len(intensity_history) >= max_len // 2:
      intensities = np.array(intensity_history)
      times = np.array(time_history)

      detrended = intensities - np.mean(intensities)
      n = len(detrended)
      dt = np.mean(np.diff(times))

      if dt > 0:
        freqs = np.fft.fftfreq(n, d=dt)
        fft_vals = np.abs(np.fft.fft(detrended))

        # Filter to valid range (0.5 Hz up to Nyquist limit ~14 Hz)
        pos_mask = (freqs > 0.5) & (freqs < (fps / 2.0))
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_vals[pos_mask]

        if len(pos_fft) > 0:
          # Find top 3 peak frequencies in the spectrum
          sorted_indices = np.argsort(pos_fft)[::-1]
          for idx in sorted_indices[:3]:
            top_peaks.append(pos_freqs[idx])

    # Overlay stats on feed
    display_frame = frame.copy()
    cv2.putText(
        display_frame,
        f'Mean Brightness: {mean_intensity:.1f}',
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        display_frame,
        f'Target Freq: {args.target_rate:.2f} Hz',
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # Display top detected harmonic peaks
    peaks_str = ', '.join([f'{p:.2f} Hz' for p in top_peaks])
    cv2.putText(
        display_frame,
        f'Top FFT Peaks: [{peaks_str}]',
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Check if target or its harmonics are present
    match_found = any(abs(p - args.target_rate) < 0.4 for p in top_peaks)
    if match_found:
      cv2.putText(
          display_frame,
          'SQUARE WAVE FOUND!',
          (30, 170),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.9,
          (0, 0, 255),
          2,
      )

    cv2.imshow('Harmonic Detector', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()