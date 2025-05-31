import time
import cv2
import os
import json
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class GhostEntity:
    def __init__(self, frames_landmarks=None):
        self.frames = frames_landmarks if frames_landmarks is not None else []
        self.display_mode = "skeleton"  # "points", "skeleton", "face_box", "body_box"
        self.start_time = time.time()  # Guardamos el tiempo en que comenzó la grabación

    def add_frame(self, landmark_list):
        self.frames.append(landmark_list)

    def draw(self, frame, frame_idx, ghost_index):
        if len(self.frames) == 0:
            return

        frame_idx %= len(self.frames)  # Repetir animación en bucle
        landmarks = self.frames[frame_idx]

        h, w, _ = frame.shape
        total_frames = len(self.frames)
        fps = 30
        elapsed_time = round((frame_idx / fps) % (total_frames / fps), 1)

        # Timer en la parte inferior
        timer_x = 50
        timer_y = h - (30 * ghost_index) - 20

        cv2.putText(frame, f"Layer {ghost_index+1} - Timer: {elapsed_time}s", 
                    (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

    def export_to_json(self, ghost_id):
        filename = f"ghost_{ghost_id}.json"
        path = os.path.join("data", filename)
        with open(path, "w") as f:
            json.dump(self.frames, f)
        print(f"✅ Fantasma exportado como {filename}")
