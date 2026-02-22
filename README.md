## 🧠 Metodología y Tecnologías Utilizadas

El proyecto fue desarrollado siguiendo la metodología **CRISP-DM**, dividiendo el trabajo en etapas claras desde la recolección de datos hasta el despliegue del prototipo.

---

### 📌 Recolección y Etiquetado del Dataset
- Grabación propia de videos en entorno controlado (cámara de celular / webcam).
- Recolección manual de videos externos (YouTube y competiciones oficiales).
- Total de **87 videos iniciales**, depurados a **72 videos etiquetados**:
  - 33 ejecuciones correctas
  - 39 ejecuciones incorrectas
- Etiquetado frame por frame para clasificación de la técnica.

---

### 🎥 Preprocesamiento de Video (CRISP-DM – Fase 3)
- **OpenCV**:
  - Recorte automático del atleta
  - Estabilización de frames
  - Reducción de ruido mediante filtro Gaussiano
  - Reducción de tasa de fotogramas a **10 FPS**
- **MediaPipe Pose**:
  - Extracción de landmarks corporales (articulaciones)
- **YOLOv8** *(uso parcial)*:
  - Detección de bounding box del atleta
- Normalización de salida:
  - Resolución **720p (1280×720)**

---

### 🤖 Modelado y Entrenamiento (CRISP-DM – Fases 4 y 5)
- Lenguaje principal: **Python**
- Modelos implementados y comparados:
  - **SVM** (scikit-learn)
  - **Random Forest** (scikit-learn)
  - **CNN – Red Neuronal Convolucional** (Keras + TensorFlow)
- Arquitectura CNN:
  - 12 capas convolucionales
  - Dropout: 0.5
  - Optimizador: Adam
  - Learning rate: 0.0001
  - Batch size: 16
  - Epochs: 30 + Early Stopping
- Validación:
  - **K-Fold Cross Validation (k = 5)**

📊 **Mejor modelo**: CNN  
📈 **Exactitud alcanzada**: **93.3 %**

---

### 🚀 Despliegue y Prototipo (CRISP-DM – Fase 6)
- **Flask**:
  - Backend API para recepción de videos y retorno de resultados
- **Streamlit**:
  - Interfaz web para carga y análisis de videos
- Modelo exportado en formato **TensorFlow SavedModel**
- Inferencia:
  - **Menor a 3 segundos por video** en CPU estándar
