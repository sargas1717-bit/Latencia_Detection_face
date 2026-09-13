import os
import urllib.request

def descargar_archivo(url, destino, nombre_descriptivo):
    if not os.path.exists(destino):
        print(f"[-] Descargando {nombre_descriptivo} ({destino})...")
        try:
            urllib.request.urlretrieve(url, destino)
            print(f"[+] {nombre_descriptivo} descargado correctamente.\n")
        except Exception as e:
            print(f"[x] Error al descargar {nombre_descriptivo}: {e}\n")
    else:
        print(f"[v] {nombre_descriptivo} ya existe ({destino}). Omitiendo descarga.\n")

if __name__ == "__main__":
    print("=" * 50)
    print(" GESTOR DE DESCARGAS DE MODELOS ")
    print("=" * 50)

    # 1. OpenCV Haar Cascades
    descargar_archivo(
        url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        destino="haarcascade_frontalface_default.xml",
        nombre_descriptivo="Clasificador Haar (OpenCV)"
    )

    # 2. MediaPipe (BlazeFace Short Range)
    descargar_archivo(
        url="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        destino="blaze_face_short_range.tflite",
        nombre_descriptivo="BlazeFace TFLite (MediaPipe)"
    )

    # 3. OpenCV YuNet (ONNX)
    descargar_archivo(
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        destino="face_detection_yunet_2023mar.onnx",
        nombre_descriptivo="YuNet ONNX (OpenCV)"
    )

    # 4. YOLOv8n-Face (HuggingFace)
    descargar_archivo(
        url="https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
        destino="face_yolov8n.pt",
        nombre_descriptivo="YOLOv8n-Face (Ultralytics)"
    )

    print("=" * 50)
    print(" Proceso de verificación/descarga completado.")
    print("=" * 50)