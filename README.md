# 🎯 Benchmark de Detección Facial de Bajo Cómputo (CPU)

Este proyecto proporciona un entorno de pruebas robusto para evaluar y comparar el rendimiento de diferentes algoritmos de detección facial (MediaPipe, YOLOv8, OpenCV YuNet y Haar Cascades) ejecutándose **exclusivamente en CPU**. 

Está diseñado específicamente para entornos de hardware limitado (como Raspberry Pi, SBCs o equipos con <= 4GB de RAM), priorizando la relación entre FPS (latencia) y el consumo de recursos computacionales.

## 📊 Algoritmos Evaluados

| Algoritmo | Descripción | Enfoque |
| :--- | :--- | :--- |
| **MediaPipe (BlazeFace)** | Solución optimizada de Google. | Extrema velocidad y bajo uso de CPU. Ideal para tiempo real. |
| **YuNet (OpenCV DNN)** | Red neuronal convolucional súper ligera exportada en ONNX. | Equilibrio sólido entre precisión y rendimiento. |
| **YOLOv8 Nano (Ultralytics)** | El modelo más pequeño de la familia YOLO. | Alta precisión, pero alto consumo de CPU debido a operaciones PyTorch. |
| **Haar Cascades (OpenCV)** | Algoritmo clásico de detección. | Bajo consumo, pero alta tasa de falsos positivos frente a rotaciones. |

## 🚀 Requisitos Previos

*   Python 3.10 o superior.
*   Cámara web funcional.
*   Entorno virtual recomendado (`venv`).

## ⚙️ Instalación y Uso

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/sargas1717-bit/Latencia_Detection_face.git
    cd Latencia_Detection_face
    ```

2.  **Crear y activar el entorno virtual:**
    ```bash
    python -m venv .venv
    # En Windows:
    .venv\Scripts\activate
    # En Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar los modelos (pesos preentrenados):**
    Este script descargará los archivos `.tflite`, `.xml`, `.onnx` y `.pt` necesarios de forma automática, manteniéndolos fuera del control de versiones.
    ```bash
    python descargar_modelos.py
    ```

5.  **Ejecutar el Benchmark:**
    ```bash
    python benchmark_rostros.py
    ```

## 📈 Exportación de Resultados
Al finalizar la evaluación de todos los modelos, el script genera automáticamente un reporte en formato `.csv` dentro de la carpeta `reportes/`. Este archivo incluye métricas detalladas de FPS, latencia, y uso promedio de CPU y RAM para cada algoritmo.

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si deseas agregar un nuevo algoritmo al benchmark o mejorar la toma de métricas, no dudes en hacer un fork y enviar un Pull Request.