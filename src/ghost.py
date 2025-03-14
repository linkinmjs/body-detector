import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class GhostEntity:
    """Clase para almacenar y dibujar la trayectoria de un 'fantasma'."""
    
    def __init__(self, frames_landmarks):
        self.frames_landmarks = frames_landmarks
        self.display_mode = "skeleton"  # "points", "skeleton", "face_box", "body_box"

    def draw(self, frame, frame_idx):
        """Dibuja la entidad fantasma en la imagen actual."""
        if len(self.frames_landmarks) == 0:
            return

        frame_idx %= len(self.frames_landmarks)
        landmarks = self.frames_landmarks[frame_idx]

        h, w, _ = frame.shape

        if self.display_mode == "points":
            for point in landmarks:
                cv2.circle(frame, point, 5, (255, 0, 0), -1)

        elif self.display_mode == "skeleton":
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            for x, y in landmarks:
                landmark = landmark_list.landmark.add()
                landmark.x = x / w
                landmark.y = y / h
                landmark.z = 0
                landmark.visibility = 1

            mp_drawing.draw_landmarks(frame, landmark_list, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        elif self.display_mode == "face_box":
            face_points = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            if all(i < len(landmarks) for i in face_points):
                x_coords = [landmarks[i][0] for i in face_points]
                y_coords = [landmarks[i][1] for i in face_points]
                cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (0, 255, 0), 2)

        elif self.display_mode == "body_box":
            body_points = [11, 12, 19, 20, 23, 24]
            if all(i < len(landmarks) for i in body_points):
                x_coords = [landmarks[i][0] for i in body_points]
                y_coords = [landmarks[i][1] for i in body_points]
                cv2.rectangle(frame, (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords)), (255, 255, 0), 2)
