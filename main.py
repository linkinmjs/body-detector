import cv2
from src.detector import PoseDetector
from src.ghost import GhostEntity
from src.main_layer_recorder import MainLayerRecorder

# Inicializar cámara
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar módulos
pose_detector = PoseDetector()
main_layer = None
ghost_entities = []
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de pose
    results = pose_detector.detect(frame)

    # Obtener dimensiones y landmarks
    h, w, _ = frame.shape
    landmarks_current = []

    if results.pose_landmarks:
        pose_detector.draw_landmarks(frame, results)

        # Convertir landmarks a píxeles
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_current.append((cx, cy))

    # Si se está grabando la capa principal, guardar frame
    if main_layer:
        main_layer.write_frame(frame)
        main_layer.draw_status(frame)

    # Dibujar fantasmas (si hay alguno)
    for i, ghost in enumerate(ghost_entities):
        ghost.draw(frame, frame_counter, i)

    frame_counter += 1

    # Mostrar ventana principal
    cv2.imshow('Pose Detector - Main Layer', frame)

    # Control de teclas
    key = cv2.waitKey(1) & 0xFF

    # Salir
    if key == ord('q'):
        break

    # Espacio → iniciar o pausar grabación
    elif key == ord(' '):
        if main_layer is None:
            main_layer = MainLayerRecorder((frame_width, frame_height))
            main_layer.toggle()  # comenzar grabación
        elif not main_layer.is_finished():
            main_layer.toggle()  # pausar o continuar

    # Enter → finalizar y exportar capa principal
    elif key == 13:  # Enter
        if main_layer and main_layer.is_paused():
            main_layer.stop_and_finalize()
            # 🔜 Reproducir el video + overlay del fantasma (lo hacemos en el próximo paso)

# Finalizar
cap.release()
cv2.destroyAllWindows()
