import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import threading
import time

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils

# Configuración de video
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para grabación de video

# Variables de grabación
recording = False
video_writer = None
frame_counter = 0  # Control de animación de los fantasmas
ghost_entities = []  # Lista de entidades fantasma

# 🎭 Clase para almacenar los movimientos de una grabación anterior


# Función para grabar video en un hilo separado
def record_video(video_writer, cap):
    while recording:
        ret, frame = cap.read()
        if ret:
            video_writer.write(frame)
        time.sleep(0.03)  # Pequeña pausa para evitar saturación de CPU


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir imagen a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Obtener dimensiones del frame
    h, w, _ = frame.shape
    landmarks_current = []

    if results.pose_landmarks:
        # Dibujar landmarks actuales en la imagen
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Guardar coordenadas de landmarks actuales
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_current.append((cx, cy))

        # Si está grabando, almacenar la trayectoria del esqueleto actual
        if recording:
            ghost_entities[-1].frames_landmarks.append(landmarks_current.copy())

    # Dibujar las entidades fantasma acumuladas con la misma apariencia de MediaPipe
    for ghost in ghost_entities:
        ghost.draw(frame, frame_counter)

    frame_counter += 1  # Control de animación de los fantasmas

    # Mostrar imagen
    cv2.imshow('Pose Detection with Ghosts Matching MediaPipe Style', frame)

    # Manejo del teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir con 'q'
        break
    elif key == ord('1'):
        for ghost in ghost_entities:
            ghost.display_mode = "points"
    elif key == ord('2'):
        for ghost in ghost_entities:
            ghost.display_mode = "skeleton"
    elif key == ord('3'):
        for ghost in ghost_entities:
            ghost.display_mode = "face_box"
    elif key == ord('4'):
        for ghost in ghost_entities:
            ghost.display_mode = "body_box"
    elif key == ord(' '):  # Iniciar/detener grabación con 'espacio'
        if not recording:
            # Iniciar nueva grabación y crear una nueva entidad fantasma
            video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            recording = True
            ghost_entities.append(GhostEntity([]))  # Nueva entidad fantasma
            thread = threading.Thread(target=record_video, args=(video_writer, cap))
            thread.start()
        else:
            # Detener grabación y guardar la trayectoria de landmarks como "fantasma"
            recording = False
            video_writer.release()
            frame_counter = 0  # Reiniciar la animación de los landmarks

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
