# Powerlifting Technique Analysis using Machine Learning

Sistema de análisis automático de la ejecución de ejercicios de powerlifting
mediante **visión por computador y redes neuronales**, enfocado en la detección
de errores técnicos y la prevención de lesiones.

---

## 🚀 Descripción del Proyecto

Este proyecto implementa un modelo de **Machine Learning en Python** que analiza
videos de ejercicios de powerlifting (sentadilla, press de banca y peso muerto)
para evaluar la correcta ejecución de la técnica.

Se aplicó la metodología **CRISP-DM**, cubriendo desde la recolección de datos
hasta el despliegue de un prototipo funcional.

---

## 🧠 Tecnologías Utilizadas

- Python
- OpenCV
- MediaPipe Pose
- TensorFlow
- Keras
- scikit-learn
- YOLOv8 (uso parcial)
- Flask
- Streamlit

---

## ⚙️ Flujo de Funcionamiento

1. Entrada de video (grabado o en tiempo real).
2. Preprocesamiento del video (recorte, estabilización, 10 FPS, 720p).
3. Extracción de landmarks corporales.
4. Entrenamiento y evaluación de modelos ML.
5. Clasificación de la ejecución como correcta o incorrecta.
6. Visualización de resultados en interfaz web.

---

## 📊 Modelado y Resultados

- Modelos evaluados: SVM, Random Forest y CNN.
- Mejor modelo: **CNN (Keras + TensorFlow)**.
- Exactitud alcanzada: **93.3 %**.
- Validación: K-Fold Cross Validation (k=5).

---

## 🚀 Despliegue

- Backend: Flask (API REST).
- Frontend: Streamlit.
- Inferencia en tiempo real (< 3 segundos por video en CPU).

---

## 🛠️ Estado del Proyecto
🟡 Prototipo funcional / En mejora continua.

---

## 👤 Autor

**Johan David Toro Ortiz**  
Ingeniero de Sistemas  
Desarrollador Backend / Python Junior  
📧 davidortiz634@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/david-ortiz-ba76953a6/
