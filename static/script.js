function uploadVideo() {
    const videoInput = document.getElementById('videoInput');
    const videoPreview = document.getElementById('videoPreview');
    const feedbackDiv = document.getElementById('feedback');
    const progressDiv = document.getElementById('progress');
    const loadingDiv = document.getElementById('loading');
    
    if (!videoInput.files.length) {
        feedbackDiv.innerText = "Por favor, selecciona un video.";
        return;
    }
    
    const file = videoInput.files[0];
    
    // Intentar mostrar el video subido
    const videoURL = URL.createObjectURL(file);
    videoPreview.src = videoURL;
    videoPreview.load();
    videoPreview.style.display = 'block';
    
    // Manejar errores de reproducción
    videoPreview.onerror = () => {
        feedbackDiv.innerText = "Advertencia: No se pudo reproducir el video. El análisis continuará con el archivo subido.";
        videoPreview.style.display = 'none';
    };
    
    // Mostrar el spinner de carga
    feedbackDiv.innerText = "";
    progressDiv.innerText = "";
    loadingDiv.style.display = 'flex';
    
    const formData = new FormData();
    formData.append('video', file);
    
    // Simular progreso
    let simulatedProgress = 0;
    const progressInterval = setInterval(() => {
        simulatedProgress += 10;
        if (simulatedProgress <= 100) {
            progressDiv.innerText = `Progreso: ${simulatedProgress}%`;
        }
    }, 1000);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(progressInterval);
        loadingDiv.style.display = 'none';
        progressDiv.innerText = "Procesamiento completado";
        if (data.error) {
            feedbackDiv.innerText = data.error;
        } else {
            feedbackDiv.innerText = data.feedback;
            // Añadir botón para descargar retroalimentación
            const downloadBtn = document.createElement('button');
            downloadBtn.innerText = 'Descargar Retroalimentación';
            downloadBtn.onclick = () => downloadFeedback(data.feedback);
            feedbackDiv.appendChild(downloadBtn);
            // Añadir botón para reiniciar
            const resetBtn = document.createElement('button');
            resetBtn.innerText = 'Subir Nuevo Video';
            resetBtn.onclick = resetInterface;
            feedbackDiv.appendChild(resetBtn);
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        loadingDiv.style.display = 'none';
        progressDiv.innerText = "Error en el procesamiento";
        feedbackDiv.innerText = "Error al procesar el video: " + error;
    });
}

function downloadFeedback(feedback) {
    const blob = new Blob([feedback], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'retroalimentacion.txt';
    a.click();
    URL.revokeObjectURL(url);
}

function resetInterface() {
    const videoPreview = document.getElementById('videoPreview');
    const feedbackDiv = document.getElementById('feedback');
    const progressDiv = document.getElementById('progress');
    videoPreview.style.display = 'none';
    videoPreview.src = '';
    feedbackDiv.innerText = '';
    progressDiv.innerText = '';
    document.getElementById('videoInput').value = '';
}