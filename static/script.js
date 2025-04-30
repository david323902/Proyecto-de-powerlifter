function uploadVideo() {
    const videoInput = document.getElementById('videoInput');
    const feedbackDiv = document.getElementById('feedback');
    const progressDiv = document.getElementById('progress');
    
    if (!videoInput.files.length) {
        feedbackDiv.innerText = "Por favor, selecciona un video.";
        return;
    }
    
    const file = videoInput.files[0];
    const formData = new FormData();
    formData.append('video', file);
    
    feedbackDiv.innerText = "Procesando video... Espere";
    progressDiv.innerText = "Progreso: 0%";
    
    // Simular progreso (esto es una simplificación, en un escenario real usarías WebSockets)
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
        progressDiv.innerText = "Procesamiento completado";
        if (data.error) {
            feedbackDiv.innerText = data.error;
        } else {
            feedbackDiv.innerText = data.feedback;
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        progressDiv.innerText = "Error en el procesamiento";
        feedbackDiv.innerText = "Error al procesar el video: " + error;
    });
}