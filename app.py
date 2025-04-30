from flask import Flask, request, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from scipy.signal import savgol_filter
import os

app = Flask(__name__)

# Configuración de la carpeta para subir videos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo Random Forest y el scaler
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    labels_df = pd.read_csv("etiquetas_powerlifting_filtrado_updated.csv")
except Exception as e:
    print(f"Error al cargar archivos: {e}")

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Función para calcular ángulos
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Función para extraer características y clasificar
def extract_features_and_classify(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, "Error: No se pudo abrir el video."
    
    predictions = []
    confidences = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        image_height, image_width, _ = frame.shape
        results = pose.process(image)
        image.flags.writeable = True
        
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            print(f"Progreso: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height]
            
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            
            features = np.array([[hip_angle, knee_angle, elbow_angle]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            predictions.append(prediction)
            confidences.append(prob[prediction])
    
    cap.release()
    
    if len(predictions) == 0:
        return None, None, None, "Error: No se detectaron puntos clave en el video."
    
    correct_ratio = np.mean(predictions)
    final_prediction = 1 if correct_ratio > 0.65 else 0
    avg_confidence = np.mean(confidences) * 100
    
    return final_prediction, avg_confidence, correct_ratio, None

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar el video
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No se seleccionó ningún video.'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún video.'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    prediction, confidence, correct_ratio, error_msg = extract_features_and_classify(video_path)
    
    if error_msg:
        os.remove(video_path)
        return jsonify({'error': error_msg})
    
    video_name = file.filename
    if "peso_muerto" in video_name:
        ejercicio = "peso_muerto"
    elif "press_banca" in video_name:
        ejercicio = "press_banca"
    elif "sentadilla" in video_name:
        ejercicio = "sentadilla"
    else:
        ejercicio = "desconocido"
    
    feedback = f"Proporción de frames correctos: {correct_ratio*100:.2f}%\n"
    if prediction == 1:
        feedback += "Movimiento correcto"
    else:
        error_detail = labels_df[(labels_df['ejercicio'] == ejercicio) & (labels_df['etiqueta'] == 'incorrecto')]['detalle_error'].iloc[0]
        feedback += f"Movimiento incorrecto: {error_detail}"
    
    feedback += f"\nConfianza promedio: {confidence:.2f}%"
    if confidence < 70:
        feedback += "\nAdvertencia: Clasificación incierta, revisa el movimiento manualmente."
    
    os.remove(video_path)
    
    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)