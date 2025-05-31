import cv2
from src.detector import PoseDetector
from src.ghost import GhostEntity
from src.main_layer_recorder import MainLayerRecorder

# --- Inicializaciones ---
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

pose_detector = PoseDetector()
main_layer = None
ghost_entities = []
frame_counter = 0

# Variables para grabar fantasmas
recording_ghost = False
current_ghost = None
ghost_counter = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_detector.detect(frame)
    h, w, _ = frame.shape
    landmarks_current = []

    if results.pose_landmarks:
        pose_detector.draw_landmarks(frame, results)

        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_current.append((cx, cy))

    # --- Grabar capa principal ---
    if main_layer:
        main_layer.write_frame(frame)
        main_layer.draw_status(frame)

    # --- Grabar capa fantasma si corresponde ---
    if recording_ghost and landmarks_current:
        current_ghost.add_frame(landmarks_current)

    # --- Dibujar fantasmas cargados ---
    for i, ghost in enumerate(ghost_entities):
        ghost.draw(frame, frame_counter, i)

    frame_counter += 1
    cv2.imshow('Pose Detector - Main Layer', frame)

    key = cv2.waitKey(1) & 0xFF

    # Salir
    if key == ord('q'):
        break

    # Grabar/pausar capa principal
    elif key == ord(' '):
        if main_layer is None:
            main_layer = MainLayerRecorder((frame_width, frame_height))
            main_layer.toggle()
        elif not main_layer.is_finished():
            main_layer.toggle()
        elif main_layer.is_finished():  # Ya terminaste la capa principal → grabar fantasmas
            if not recording_ghost:
                print(f"🎬 Iniciando grabación del fantasma {ghost_counter}")
                recording_ghost = True
                current_ghost = GhostEntity()
            else:
                print(f"✅ Fantasma {ghost_counter} finalizado y guardado.")
                recording_ghost = False
                current_ghost.export_to_json(ghost_counter)
                ghost_entities.append(current_ghost)
                current_ghost = None
                ghost_counter += 1

    # Finalizar capa principal
    elif key == 13:  # Enter
        if main_layer and main_layer.is_paused():
            main_layer.stop_and_finalize()
            print("✅ Capa principal finalizada.")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
