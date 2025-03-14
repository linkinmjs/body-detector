import cv2

class FinalVideoExporter:
    """Clase para exportar el video con todas las capas de fantasmas incluidas."""

    def __init__(self, output_filename, frame_size, fps=30):
        """Inicializa el exportador de video."""
        self.output_filename = output_filename
        self.frame_size = frame_size
        self.fps = fps
        self.video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    def write_frame(self, frame):
        """Guarda un frame en el video final."""
        self.video_writer.write(frame)

    def close(self):
        """Cierra el archivo de video."""
        self.video_writer.release()
