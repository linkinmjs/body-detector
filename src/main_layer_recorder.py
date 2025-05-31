import cv2
import os
import time

class MainLayerRecorder:
    def __init__(self, frame_size, fps=30, output_name="main_layer.mp4", codec="mp4v"):
        self.frame_size = frame_size
        self.fps = fps
        self.output_path = os.path.join("data", output_name)
        self.codec = codec

        os.makedirs("data", exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*self.codec), fps, frame_size
        )

        self.recording = False
        self.start_time = None
        self.paused_time = 0
        self.total_paused_duration = 0
        self.finished = False

    def toggle(self):
        """Inicia o pausa la grabación."""
        if self.finished:
            return

        if not self.recording:
            # Empezar a grabar
            if self.start_time is None:
                self.start_time = time.time()
            else:
                self.total_paused_duration += time.time() - self.paused_time
            self.recording = True
        else:
            # Pausar
            self.recording = False
            self.paused_time = time.time()

    def stop_and_finalize(self):
        """Finaliza la grabación y cierra el archivo."""
        if not self.finished:
            self.video_writer.release()
            self.finished = True
            print(f"✅ Video exportado: {self.output_path}")

    def write_frame(self, frame):
        """Guarda el frame en el video si se está grabando."""
        if self.recording:
            self.video_writer.write(frame)

    def draw_status(self, frame):
        """Dibuja el timer y el estado de grabación (círculo)."""
        if self.start_time is None:
            return

        h, w, _ = frame.shape

        # Círculo de estado
        color = (0, 0, 255) if self.recording else (0, 255, 0)
        cv2.circle(frame, (30, 30), 10, color, -1)

        # Timer
        if self.recording:
            elapsed = time.time() - self.start_time - self.total_paused_duration
        else:
            elapsed = self.paused_time - self.start_time - self.total_paused_duration

        timer_text = f"{elapsed:.1f}s"
        cv2.putText(frame, timer_text, (55, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def is_recording(self):
        return self.recording

    def is_paused(self):
        return not self.recording and not self.finished and self.start_time is not None

    def is_finished(self):
        return self.finished
