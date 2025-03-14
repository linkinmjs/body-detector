import cv2
import time
from src.detector import PoseDetector
from src.ghost import GhostEntity
from src.video_recorder import VideoRecorder
from src.video_exporter import FinalVideoExporter


# Configuración de la cámara
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Inicializar módulos
pose_detector = PoseDetector()
ghost_entities = []
video_recorder = None
recording = False
frame_counter = 0  # Control de animación de los fantasmas
exporter = FinalVideoExporter("final_output.avi", (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de pose
    results = pose_detector.detect(frame)

    # Obtener dimensiones del frame
    h, w, _ = frame.shape
    landmarks_current = []

    if results.pose_landmarks:
        # Dibujar landmarks actuales
        pose_detector.draw_landmarks(frame, results)

        # Guardar coordenadas de landmarks actuales
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_current.append((cx, cy))

        # Si está grabando, almacenar la trayectoria del esqueleto actual
        if recording:
            ghost_entities[-1].frames_landmarks.append(landmarks_current.copy())

    # Dibujar los fantasmas acumulados
    for i, ghost in enumerate(ghost_entities):
        ghost.draw(frame, frame_counter, i)  # Pasamos el índice del fantasma

    frame_counter += 1  # Control de animación de los fantasmas
    exporter.write_frame(frame)

    # Mostrar imagen
    cv2.imshow('Pose Detection with Configurable Ghosts', frame)

    # Manejo del teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Salir con 'q'
        break
    elif key == ord(' '):  # Iniciar/detener grabación con 'espacio'
        if not recording:
            # Iniciar grabación y agregar nuevo fantasma
            video_recorder = VideoRecorder("output.avi", (frame_width, frame_height))
            video_recorder.start_recording(cap)
            recording = True
            ghost_entities.append(GhostEntity([]))  # Nueva entidad fantasma
        else:
            # Detener grabación
            recording = False
            video_recorder.stop_recording()
            frame_counter = 0  # Reiniciar la animación de los landmarks

    # Cambiar el modo de visualización de los fantasmas
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

# Liberar recursos
cap.release()
exporter.close()
cv2.destroyAllWindows()
