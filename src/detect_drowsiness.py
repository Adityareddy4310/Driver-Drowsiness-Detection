import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import pygame
import os
from collections import deque

# ─── Reliable paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "drowsy_yolo3", "weights", "best.pt")
ALARM_PATH = os.path.join(PROJECT_ROOT, "alarm.wav")

# ─── Setup ─────────────────────────────────────────────────────────────────────
pygame.mixer.init()
alarm_sound = None
try:
    alarm_sound = pygame.mixer.Sound(ALARM_PATH)
except Exception as e:
    print(f"Warning: Could not load alarm sound → {e}")

try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Checked path: {MODEL_PATH}")
    exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not opened. Try cv2.VideoCapture(1).")
    exit(1)

# ─── Buffers & counters ────────────────────────────────────────────────────────
EAR_BUFFER_SIZE = 10          # larger buffer → more stable
MAR_BUFFER_SIZE = 5
ear_buffer = deque(maxlen=EAR_BUFFER_SIZE)
mar_buffer = deque(maxlen=MAR_BUFFER_SIZE)

LOW_EAR_COUNT = 0             # new: count consecutive low EAR frames
LOW_EAR_REQUIRED = 8          # need ~8 low frames before considering drowsy (stronger filter)

# ─── Helper functions ──────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def mouth_aspect_ratio(mouth):
    if len(mouth) != 8:
        return 0.0
    horizontal = np.linalg.norm(mouth[0] - mouth[1])
    if horizontal == 0:
        return 0.0
    v1 = np.linalg.norm(mouth[2] - mouth[3])
    v2 = np.linalg.norm(mouth[4] - mouth[6])
    v3 = np.linalg.norm(mouth[5] - mouth[7])
    avg_vertical = (v1 + v2 + v3) / 3.0
    return avg_vertical / horizontal

# Landmark indices (same as before – mouth fix already applied)
LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 0, 17, 63, 293, 67, 297]

# ─── Thresholds – tuned based on your data ─────────────────────────────────────
EAR_THRESHOLD = 0.185         # lower than your normal ~0.22–0.25; drowsy <0.185
MAR_THRESHOLD = 0.80          # your normal ~0.57–0.60; yawn >0.80
DROWSY_SECONDS = 4.5          # longer delay

drowsy_start = None

# ─── Main loop ─────────────────────────────────────────────────────────────────
print("\n=== Press 'q' to quit ===\n")
print("Your normal EAR seems ~0.22–0.28 → threshold set low.\n"
      "Test: keep eyes open wide → should stay ALERT most time.\n"
      "Close eyes long → should trigger after ~4–5 sec.\n")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # YOLO (keep for visualization)
    results = model(frame, conf=0.35, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = r.names[int(box.cls)]
            conf = float(box.conf)
            color = (0, 255, 0) if "open" in label.lower() or "awake" in label.lower() else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb)

    avg_ear = 0.0
    avg_mar = 0.0
    is_drowsy = False

    if mesh_results.multi_face_landmarks:
        for face_lms in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, face_lms, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=1)
            )

            h, w, _ = frame.shape
            lm = face_lms.landmark

            left_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
            right_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])
            ear_left = eye_aspect_ratio(left_pts)
            ear_right = eye_aspect_ratio(right_pts)
            ear = (ear_left + ear_right) / 2.0

            mouth_pts = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_POINTS])
            mar = mouth_aspect_ratio(mouth_pts)

            ear_buffer.append(ear)
            mar_buffer.append(mar)
            avg_ear = np.mean(ear_buffer) if ear_buffer else ear
            avg_mar = np.mean(mar_buffer) if mar_buffer else mar

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"MAR: {avg_mar:.3f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Low EAR counter (for eyes closed detection)
            if avg_ear < EAR_THRESHOLD:
                LOW_EAR_COUNT += 1
            else:
                LOW_EAR_COUNT = 0

            # Drowsy if low EAR persistent OR high MAR
            if LOW_EAR_COUNT >= LOW_EAR_REQUIRED or avg_mar > MAR_THRESHOLD:
                is_drowsy = True

    # Console debug
    if frame_count % 15 == 0:
        status = "DROWSY" if is_drowsy else "ALERT"
        print(f"Frame {frame_count:4d} | EAR: {avg_ear:.3f} | MAR: {avg_mar:.3f} | low_ear_count: {LOW_EAR_COUNT} | {status}")

    # Timer logic
    now = time.time()
    if is_drowsy:
        if drowsy_start is None:
            drowsy_start = now
        if now - drowsy_start >= DROWSY_SECONDS:
            cv2.putText(frame, "WAKE UP! DROWSY!", (70, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 5)
            if alarm_sound and not pygame.mixer.get_busy():
                alarm_sound.play()
    else:
        drowsy_start = None
        if pygame.mixer.get_busy():
            pygame.mixer.stop()

    cv2.imshow("Driver Drowsiness Detection – press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\nProgram ended.\n")