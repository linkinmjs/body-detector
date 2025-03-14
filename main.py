import cv2
import mediapipe as mp
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
frames_landmarks = []  # Almacenará el historial de movimiento de los landmarks previos
frame_counter = 0  # Contador de cuadros para la reproducción de los "fantasmas"

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

    # Lista de landmarks del frame actual
    landmarks_current = []

    if results.pose_landmarks:
        # Dibujar landmarks actuales en la imagen
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Guardar coordenadas de landmarks actuales
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_current.append((cx, cy))

        # Si está grabando, guardar los landmarks actuales en la lista de historial
        if recording:
            frames_landmarks.append(landmarks_current.copy())

    # Dibujar los landmarks del video anterior con animación
    if not recording and len(frames_landmarks) > 0:
        frame_counter = (frame_counter + 1) % len(frames_landmarks)  # Control de bucle
        for point in frames_landmarks[frame_counter]:  # Recorrer el historial frame a frame
            cv2.circle(frame, point, 5, (255, 0, 0), -1)  # Dibujar como "fantasmas" en azul

    # Mostrar imagen
    cv2.imshow('Pose Detection with Ghost Landmarks', frame)

    # Manejo del teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir con 'q'
        break
    elif key == ord(' '):  # Iniciar/detener grabación con 'espacio'
        if not recording:
            # Iniciar grabación
            video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            recording = True
            frames_landmarks.clear()  # Borrar el historial anterior para grabar uno nuevo
            thread = threading.Thread(target=record_video, args=(video_writer, cap))
            thread.start()
        else:
            # Detener grabación y mantener historial de movimiento de landmarks
            recording = False
            video_writer.release()
            frame_counter = 0  # Reiniciar la animación de los landmarks

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
