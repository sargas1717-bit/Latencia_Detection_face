import cv2
import time
import psutil
import os
import csv
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import supervision as sv
from tabulate import tabulate

# ==========================================================
# 1. WRAPPERS DE INFERENCIA (SIN LÓGICA DE RED)
# ==========================================================

class MediaPipeDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        return detection_result.detections if detection_result.detections else []


class OpenCVHaarDetector:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))


class OpenCVYuNetDetector:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            model="face_detection_yunet_2023mar.onnx",
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect(self, frame):
        height, width, _ = frame.shape
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []


class YOLOSupervisionDetector:
    def __init__(self):
        self.model = YOLO("face_yolov8n.pt")
        self.box_annotator = sv.BoxAnnotator()

    def detect(self, frame):
        results = self.model(frame, verbose=False, device='cpu')[0]
        return sv.Detections.from_ultralytics(results)


# ==========================================================
# 2. MOTOR DE EVALUACIÓN
# ==========================================================

def run_benchmark(model_name, detector_instance, camera_index=0, duration_sec=10):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[x] Error: No se pudo acceder a la cámara para {model_name}.")
        return None

    process = psutil.Process(os.getpid())
    
    print(f"\n[+] Cargando y estabilizando: {model_name}...")
    
    for _ in range(15):
        ret, frame = cap.read()
        if ret:
            try:
                detector_instance.detect(frame)
            except Exception as e:
                print(f"[x] Error FATAL en el modelo {model_name} durante calentamiento:")
                print(f"    Detalle: {e}")
                cap.release()
                return None 

    latencies_ms = []
    cpu_measurements = []
    ram_measurements = []

    print(f"[>] Evaluando durante {duration_sec} segundos...")
    start_time = time.time()
    process.cpu_percent()

    while (time.time() - start_time) < duration_sec:
        ret, frame = cap.read()
        if not ret:
            print("[x] Error: Se cortó la señal de la cámara.")
            break

        t0 = time.perf_counter()
        
        try:
            _ = detector_instance.detect(frame)
        except Exception as e:
            print(f"[x] Error FATAL en el modelo {model_name} durante evaluación:")
            print(f"    Detalle: {e}")
            break 

        t1 = time.perf_counter()

        latency = (t1 - t0) * 1000
        latencies_ms.append(latency)
        cpu_measurements.append(process.cpu_percent())
        ram_measurements.append(process.memory_info().rss / (1024 * 1024))

        cv2.putText(frame, f"{model_name} | {1000/latency:.1f} FPS", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Benchmark en vivo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0
    avg_cpu = sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0
    avg_ram = sum(ram_measurements) / len(ram_measurements) if ram_measurements else 0

    return {
        "Modelo": model_name,
        "FPS Promedio": round(avg_fps, 1),
        "Latencia (ms)": round(avg_latency, 2),
        "CPU Promedio (%)": round(avg_cpu, 1),
        "RAM Promedio (MB)": round(avg_ram, 1)
    }

# ==========================================================
# 3. EJECUCIÓN Y EXPORTACIÓN
# ==========================================================

def exportar_csv(resultados):
    if not resultados:
        print("\n[!] No hay resultados para exportar.")
        return

    # Crear carpeta 'reportes' si no existe
    if not os.path.exists("reportes"):
        os.makedirs("reportes")

    # Generar nombre de archivo con fecha y hora actual (ej: benchmark_20260912_225309.csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"reportes/benchmark_{timestamp}.csv"

    # Extraer los nombres de las columnas desde el primer diccionario
    columnas = resultados[0].keys()

    try:
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
            escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
            escritor.writeheader()
            for fila in resultados:
                escritor.writerow(fila)
        print(f"\n[+] Resultados exportados exitosamente a: {nombre_archivo}")
    except Exception as e:
        print(f"\n[x] Error al exportar el archivo CSV: {e}")

if __name__ == "__main__":
    if not os.path.exists("blaze_face_short_range.tflite"):
        print("Error: Faltan archivos. Ejecuta primero 'python descargar_modelos.py'")
        exit()

    tests = [
        ("OpenCV (Haar Cascade)", OpenCVHaarDetector()),
        ("MediaPipe (BlazeFace)", MediaPipeDetector()),
        ("OpenCV (YuNet ONNX)", OpenCVYuNetDetector()),
        ("YOLOv8 Nano + Supervision", YOLOSupervisionDetector())
    ]

    results_table = []
    
    print("=" * 60)
    print(" INICIO DEL BENCHMARK COMPARATIVO")
    print("=" * 60)

    for name, detector in tests:
        stats = run_benchmark(name, detector, camera_index=0, duration_sec=10)
        if stats:
            results_table.append(stats)
        time.sleep(2)

    print("\n" + "=" * 60)
    print(" RESULTADOS FINALES")
    print("=" * 60)
    print(tabulate(results_table, headers="keys", tablefmt="fancy_grid"))
    
    # Llamada a la nueva función
    exportar_csv(results_table)