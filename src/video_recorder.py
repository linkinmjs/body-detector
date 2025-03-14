import cv2
import time
import threading

class VideoRecorder:
    """Clase para manejar la grabación de video en un hilo separado."""

    def __init__(self, output_filename, frame_size):
        self.video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), 30.0, frame_size)
        self.recording = False

    def start_recording(self, cap):
        self.recording = True
        thread = threading.Thread(target=self._record, args=(cap,))
        thread.start()

    def _record(self, cap):
        while self.recording:
            ret, frame = cap.read()
            if ret:
                self.video_writer.write(frame)
            time.sleep(0.03)

    def stop_recording(self):
        self.recording = False
        self.video_writer.release()
