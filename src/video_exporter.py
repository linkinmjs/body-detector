import cv2
import os

class FinalVideoExporter:
    """Clase para exportar el video con todas las capas de fantasmas incluidas."""

    def __init__(self, output_filename, frame_size, fps=30, codec="XVID"):
        """Inicializa el exportador de video."""
        self.frame_size = frame_size
        self.fps = fps
        self.codec = codec

        # Asegurar que la carpeta 'data/' existe
        os.makedirs("data", exist_ok=True)

        # Ruta completa del archivo de salida dentro de 'data/'
        self.output_path = os.path.join("data", output_filename)

        # Crear el objeto VideoWriter
        self.video_writer = cv2.VideoWriter(
            self.output_path, cv2.VideoWriter_fourcc(*self.codec), fps, frame_size
        )

    def write_frame(self, frame):
        """Guarda un frame en el video final."""
        self.video_writer.write(frame)

    def close(self):
        """Cierra el archivo de video."""
        self.video_writer.release()
        print(f"✅ Video exportado correctamente en: {self.output_path}")
