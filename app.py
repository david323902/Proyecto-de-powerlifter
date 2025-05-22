from flask import Flask, request, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Configuración de la carpeta para subir videos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Variable global para almacenar el DataFrame de etiquetas
labels_df = None

# Cargar el modelo Random Forest y el scaler
print("Cargando archivos...")
try:
    model = joblib.load("random_forest_model.joblib")
    print("Modelo Random Forest cargado correctamente.")
    scaler = joblib.load("scaler.joblib")
    print("Scaler cargado correctamente.")
    
    # Cargar y procesar el CSV de etiquetas
    labels_df = pd.read_csv("etiquetas_powerlifting_filtrado_updated.csv", encoding='utf-8')
    # Limpiar espacios en blanco y normalizar los datos
    labels_df['ejercicio'] = labels_df['ejercicio'].str.strip().str.lower()
    labels_df['etiqueta'] = labels_df['etiqueta'].str.strip().str.lower()
    
    print(f"Archivo de etiquetas cargado. Filas: {len(labels_df)}")
    print(f"Columnas disponibles: {labels_df.columns.tolist()}")
    print(f"Ejercicios únicos: {labels_df['ejercicio'].unique()}")
    print(f"Etiquetas únicas: {labels_df['etiqueta'].unique()}")
    
    # Verificar si hay datos de peso muerto incorrectos
    example = labels_df[(labels_df['ejercicio'] == 'peso_muerto') & (labels_df['etiqueta'] == 'incorrecto')]
    print(f"Ejemplo de peso_muerto incorrecto: {len(example)} filas")
    if not example.empty:
        print(example.iloc[0])
except Exception as e:
    print(f"Error al cargar archivos: {e}")
    raise

# Inicializar MediaPipe
print("Inicializando MediaPipe...")
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("MediaPipe inicializado correctamente.")
except Exception as e:
    print(f"Error al inicializar MediaPipe: {e}")
    raise

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

# Normalizar nombre del ejercicio
def normalize_exercise_name(video_name):
    video_name = video_name.lower()
    if "peso_muerto" in video_name or "peso muerto" in video_name or "deadlift" in video_name:
        return "peso_muerto"
    elif "press_banca" in video_name or "press banca" in video_name or "bench press" in video_name:
        return "press_banca"
    elif "sentadilla" in video_name or "squat" in video_name:
        return "sentadilla"
    else:
        return "desconocido"

# Función para extraer características y clasificar
def extract_features_and_classify(video_path):
    print(f"Procesando video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, "Error: No se pudo abrir el video."
    
    predictions = []
    confidences = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Procesar cada 5 frames para acelerar
    frame_interval = 5
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        
        frame = cv2.resize(frame, (640, 480))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        image_height, image_width, _ = frame.shape
        results = pose.process(image)
        image.flags.writeable = True
        
        if frame_count % (100 // frame_interval) == 0 or frame_count >= total_frames:
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

# Función para obtener el detalle del error según el ejercicio
def get_error_detail(ejercicio):
    global labels_df
    
    print(f"Buscando detalles de error para: '{ejercicio}' con etiqueta 'incorrecto'")
    
    # Verificar si el DataFrame está disponible
    if labels_df is None or labels_df.empty:
        print("Error: DataFrame de etiquetas vacío o no disponible")
        return "Detalles del error no disponibles (CSV no cargado correctamente)."
    
    # Verificar si las columnas necesarias existen
    if 'ejercicio' not in labels_df.columns or 'etiqueta' not in labels_df.columns or 'detalle_error' not in labels_df.columns:
        print(f"Error: Columnas faltantes en el CSV. Columnas disponibles: {labels_df.columns.tolist()}")
        return "Detalles del error no disponibles (formato de CSV incorrecto)."
    
    # Filtrar el DataFrame
    filtered_df = labels_df[(labels_df['ejercicio'] == ejercicio) & (labels_df['etiqueta'] == 'incorrecto')]
    print(f"Filas encontradas para {ejercicio} incorrecto: {len(filtered_df)}")
    
    # Si hay resultados, devolver el detalle del primer error
    if not filtered_df.empty:
        detail = filtered_df['detalle_error'].iloc[0]
        print(f"Detalle de error encontrado: {detail}")
        return detail
    else:
        print(f"No se encontraron detalles de error para {ejercicio}")
        return f"Detalles del error no disponibles para {ejercicio}."

# Ruta principal
@app.route('/')
def index():
    print("Accediendo a la ruta principal...")
    return render_template('index.html')

# Ruta para procesar el video
@app.route('/upload', methods=['POST'])
def upload_video():
    print("Recibiendo solicitud de carga de video...")
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
    
    # Normalizar el nombre del ejercicio
    ejercicio = normalize_exercise_name(file.filename)
    print(f"Ejercicio detectado: {ejercicio}")
    
    # Generar feedback
    feedback_text = f"Proporción de frames correctos: {correct_ratio*100:.2f}%\n"
    if prediction == 1:
        feedback_text += "Movimiento correcto"
    else:
        # Obtener detalles del error
        error_detail = get_error_detail(ejercicio)
        feedback_text += f"Movimiento incorrecto: {error_detail}"
    
    feedback_text += f"\nConfianza promedio: {confidence:.2f}%"
    if confidence < 70:
        feedback_text += "\nAdvertencia: Clasificación incierta, revisa el movimiento manualmente."
    
    # Preparar datos para Highcharts (pie chart)
    chart_data = {
        'title': {'text': f'Resultados de {ejercicio}'},
        'chart': {'type': 'pie'},
        'series': [{
            'name': 'Porcentajes',
            'data': [
                {'name': 'Frames Correctos', 'y': correct_ratio * 100},
                {'name': 'Confianza', 'y': confidence}
            ]
        }]
    }
    
    os.remove(video_path)
    
    return jsonify({'feedback': feedback_text, 'chart_data': chart_data})

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(debug=False, host='0.0.0.0', port=5000)


