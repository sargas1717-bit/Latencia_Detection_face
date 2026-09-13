import cv2
import math
import time
import psutil
import os
import csv
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================================
# ⚙️ CONFIGURACIÓN DEL SISTEMA (PANEL DE CONTROL)
# ==========================================================
MOSTRAR_BOUNDING_BOX  = True
MOSTRAR_CENTROIDE     = True
MOSTRAR_ETIQUETAS     = True
MOSTRAR_DISTANCIA     = True   # NUEVO: Muestra la distancia estimada en cm
EXPORTAR_METRICAS     = True

MAX_DISTANCIA_TRACKER = 50
DURACION_PRUEBA_SEC   = 20

# Constantes de calibración óptica (Aproximadas)
ANCHO_REAL_ROSTRO_CM  = 15.0   # Ancho promedio de un rostro humano
DISTANCIA_FOCAL_APROX = 600    # Valor típico para webcams a 640x480

# ==========================================================
# 1. CLASE DE TRACKING
# ==========================================================
class RastreadorLigero:
    def __init__(self, max_distancia=50):
        self.centroides = {}
        self.id_actual = 0
        self.max_distancia = max_distancia

    def actualizar(self, detecciones_rects):
        objetos_rastreados = []
        nuevos_centroides = {}

        for rect in detecciones_rects:
            x, y, w, h = rect
            cx = x + w // 2
            cy = y + h // 2

            mismo_objeto_detectado = False
            
            for obj_id, pt in self.centroides.items():
                distancia = math.hypot(cx - pt[0], cy - pt[1])
                
                if distancia < self.max_distancia:
                    self.centroides[obj_id] = (cx, cy)
                    objetos_rastreados.append((x, y, w, h, obj_id, cx, cy))
                    mismo_objeto_detectado = True
                    break

            if not mismo_objeto_detectado:
                self.centroides[self.id_actual] = (cx, cy)
                objetos_rastreados.append((x, y, w, h, self.id_actual, cx, cy))
                self.id_actual += 1
                
        for obj in objetos_rastreados:
            _, _, _, _, obj_id, cx, cy = obj
            nuevos_centroides[obj_id] = (cx, cy)
        
        self.centroides = nuevos_centroides
        return objetos_rastreados

# ==========================================================
# 2. INICIALIZACIÓN DEL SISTEMA
# ==========================================================
def main():
    print("=" * 50)
    print(" INICIANDO TRACKER LIGERO + PROFUNDIDAD")
    print("=" * 50)

    rastreador = RastreadorLigero(max_distancia=MAX_DISTANCIA_TRACKER)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    proceso = psutil.Process(os.getpid())
    proceso.cpu_percent()

    metricas_registro = []
    tiempo_inicio = time.time()
    
    print("[>] Presiona 'q' en la ventana de video para salir.")

    while True:
        tiempo_actual = time.time()
        if DURACION_PRUEBA_SEC > 0 and (tiempo_actual - tiempo_inicio) > DURACION_PRUEBA_SEC:
            print("[i] Tiempo de prueba finalizado.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        resultados = detector.detect(mp_image)
        
        t1 = time.perf_counter()

        rects_detectados = []
        if resultados.detections:
            for det in resultados.detections:
                bbox = det.bounding_box
                rects_detectados.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))

        rostros_rastreados = rastreador.actualizar(rects_detectados)

        # Variables para exportación
        distancia_promedio_frame = 0
        total_distancias = 0

        for rostro in rostros_rastreados:
            x, y, w, h, obj_id, cx, cy = rostro

            # --- CÁLCULO DE PROFUNDIDAD ---
            # Evitar división por cero si la caja tiene ancho 0 por algún bug de lectura
            if w > 0:
                distancia_cm = (ANCHO_REAL_ROSTRO_CM * DISTANCIA_FOCAL_APROX) / w
                distancia_promedio_frame += distancia_cm
                total_distancias += 1
            else:
                distancia_cm = 0

            # --- DIBUJADO DE INTERFAZ ---
            if MOSTRAR_BOUNDING_BOX:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 2)
            
            if MOSTRAR_CENTROIDE:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            if MOSTRAR_ETIQUETAS:
                etiqueta = f"Face_{obj_id}"
                cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            if MOSTRAR_DISTANCIA and distancia_cm > 0:
                cv2.putText(frame, f"{int(distancia_cm)} cm", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        latencia_ms = (t1 - t0) * 1000
        fps = 1000 / latencia_ms if latencia_ms > 0 else 0
        cpu_uso = proceso.cpu_percent()
        ram_uso = proceso.memory_info().rss / (1024 * 1024)
        
        dist_exportar = round(distancia_promedio_frame / total_distancias, 1) if total_distancias > 0 else 0.0

        if EXPORTAR_METRICAS:
            metricas_registro.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "Rostros_Detectados": len(rostros_rastreados),
                "Distancia_Aprox_cm": dist_exportar,
                "FPS": round(fps, 1),
                "Latencia_ms": round(latencia_ms, 2),
                "CPU_Percent": round(cpu_uso, 1),
                "RAM_MB": round(ram_uso, 1)
            })

        cv2.putText(frame, f"FPS: {fps:.1f} | CPU: {cpu_uso:.1f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Tracker Ligero - MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[i] Abortado por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if EXPORTAR_METRICAS and metricas_registro:
        if not os.path.exists("reportes"):
            os.makedirs("reportes")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"reportes/tracking_metricas_{timestamp}.csv"
        
        with open(nombre_archivo, 'w', newline='', encoding='utf-8') as archivo_csv:
            columnas = metricas_registro[0].keys()
            escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(metricas_registro)
            
        print(f"[+] Reporte exportado a: {nombre_archivo}")

if __name__ == "__main__":
    main()