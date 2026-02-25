# Análisis Automático de Técnica en Powerlifting con Inteligencia Artificial

## 📌 Descripción
Proyecto de **Machine Learning y Visión por Computador** orientado a la **detección automática de errores en la ejecución de ejercicios de powerlifting** (sentadilla, press de banca y peso muerto), utilizando análisis de video y redes neuronales.

El sistema evalúa la técnica del atleta y clasifica la ejecución como **correcta o incorrecta**, proporcionando una base para retroalimentación automática y prevención de lesiones.

## 🎯 Objetivo del Proyecto
Desarrollar un modelo de IA capaz de:
- Analizar movimientos complejos y multiarticulares
- Identificar patrones biomecánicos incorrectos
- Apoyar la mejora del rendimiento y reducción de lesiones

## 🚀 Tecnologías Utilizadas

### Procesamiento de Video
- **OpenCV** – Procesamiento de imágenes y video
- **MediaPipe Pose** – Detección de puntos clave del cuerpo (landmarks)
- **YOLOv8** – Detección del atleta (bounding box, opcional)
- **Filtro Gaussiano** – Reducción de ruido
- **Normalización de video** – 720p, 10 FPS

### Machine Learning / Deep Learning
- **Python** – Lenguaje principal
- **TensorFlow** – Framework de Deep Learning
- **Keras** – Construcción y entrenamiento de CNN
- **scikit-learn** – Modelos clásicos (SVM, Random Forest)
- **k-fold cross-validation (k=5)** – Validación del modelo

### Modelos Implementados
- **SVM**
- **Random Forest**
- **CNN (Red Neuronal Convolucional)**  
  - 12 capas convolucionales  
  - Dropout: 0.5  
  - Optimizador Adam  
  - Learning rate: 0.0001  
  - Batch size: 16  
  - Early stopping  

### Resultados
- **Exactitud máxima: 93.3 %** (CNN)

### Despliegue y Prototipo
- **Flask** – Backend para inferencia
- **Streamlit** – Interfaz web
- **TensorFlow SavedModel** – Modelo optimizado
- **Inferencia en tiempo real (< 3 segundos en CPU)**

## ⚙️ Flujo del Sistema
1. Carga o grabación del video
2. Preprocesamiento automático
3. Extracción de landmarks corporales
4. Inferencia del modelo de IA
5. Clasificación de la técnica
6. Visualización del resultado

## 🧠 Mi Rol en el Proyecto
- Diseño del pipeline completo de IA
- Recolección y etiquetado del dataset
- Preprocesamiento de videos
- Entrenamiento y evaluación de modelos
- Desarrollo del backend y prototipo funcional

## 📊 Dataset
- 87 videos iniciales
- 72 videos finales etiquetados
- 33 ejecuciones correctas
- 39 ejecuciones incorrectas
- Videos propios y de competiciones oficiales

## 📌 Estado del Proyecto
🟢 Prototipo funcional  
🔧 En mejora continua

## 📚 Aprendizajes Clave
- Visión por computador aplicada al deporte
- Machine Learning con datos reales
- Análisis de movimientos complejos
- Despliegue de modelos de IA
- Integración IA + aplicaciones web

## 👤 Autor

**Johan David Toro Ortiz**  
Ingeniero de Sistemas  
Desarrollador Backend / Python Junior  
📧 davidortiz634@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/david-ortiz-ba76953a6/
