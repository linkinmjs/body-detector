import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Importamos la utilidad para dibujar

class PoseDetector:
    """Módulo de detección de poses usando MediaPipe."""
    
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    
    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)

    def draw_landmarks(self, frame, results):
        """Dibuja los landmarks en el frame si existen."""
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
