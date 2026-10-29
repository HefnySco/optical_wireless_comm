# Optical Wireless Communication (OWC) & Software-Defined OCC POC

Welcome to the documentation of an experimental journey exploring **Optical Wireless Communication (OWC)**, **Pulse Modulation**, and **Optical Camera Communication (OCC)** using Python, OpenCV, and NumPy.

What started as a question about using a laser beam to transmit sound via AM/FM modulation evolved into a fully functional software-defined Proof of Concept (POC) using a computer monitor as a transmitter and a webcam as a receiver!

---

## 🚀 Project Overview & Evolution

1. **The Laser & Audio Idea:** We explored how a continuous laser or LED can carry audio signals using Amplitude Modulation (Intensity Modulation) or Pulse Modulation, paired with a high-speed phototransistor and amplifier.
2. **The Microcontroller Angle:** Discussed how an **STM32F411** microcontroller and a red LED can handle high-speed PWM or digital PCM pulses to cut through ambient light noise.
3. **The Software POC (Monitor & Webcam):** Pivoted to a creative software simulation where a computer screen acts as a pulsing optical transmitter, and a webcam (`/dev/video0`) acts as an ambient light sensor utilizing Fast Fourier Transform (FFT) frequency analysis.

---

## 🛠️ The Python Scripts

This project consists of two independent Python scripts that communicate optically through the room's ambient lighting changes.

### Prerequisites

Make sure you have the required Python libraries installed:

```bash
pip install opencv-python numpy

```

---

### 1. The Transmitter (`transmitter.py`)

This script opens a full-screen window, draws a custom-sized rectangle, and pulses its brightness sharply using a **Square Wave** at a specified frequency and RGB color.

```python
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
      help='Pulse frequency in Hz (e.g., 7.0)',
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

```

---

### 2. The Harmonic Receiver / Detector (`detector.py`)

This script captures frames from `/dev/video0`, calculates the average frame brightness history, runs an **FFT (Fast Fourier Transform)**, and extracts the top frequency peaks to detect the square wave and its harmonics.

```python
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
      default=7.0,
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

        pos_mask = (freqs > 0.5) & (freqs < (fps / 2.0))
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_vals[pos_mask]

        if len(pos_fft) > 0:
          sorted_indices = np.argsort(pos_fft)[::-1]
          for idx in sorted_indices[:3]:
            top_peaks.append(pos_freqs[idx])

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

```

---

## 🧪 How to Run the Experiment

1. Open **Terminal 1** and start the transmitter at **7 Hz** (which hits the sweet spot of camera sampling and harmonic isolation):
```bash
python transmitter.py --rate 7.0 --r 255 --g 255 --b 255

```


2. Open **Terminal 2** and start the detector, pointing your webcam toward the room wall/surface illuminated by the screen:
```bash
python detector.py --target-rate 7.0

```


3. Watch the live HUD: the detector will isolate the frequency peaks and instantly display **`SQUARE WAVE FOUND!`** when locked on.

---

## 💡 Key Takeaways & Physics Learnings

* **The Nyquist Limit:** Standard webcams run at ~30 FPS, meaning the maximum frequency they can reliably reconstruct without aliasing is around 15 Hz.
* **Square Waves & Harmonics:** Unlike smooth sine waves, square waves contain rich odd harmonics ($f, 3f, 5f\dots$).
* **The Sweet Spot (7 Hz):** Running at 7 Hz gives a fundamental frequency that is high enough to be distinct from slow ambient light drift, yet low enough that its primary components stay safely below the camera's 15 Hz Nyquist ceiling—enabling **exact, pin-point frequency detection**!