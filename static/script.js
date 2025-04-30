function uploadVideo() {
    const videoInput = document.getElementById('videoInput');
    const feedbackDiv = document.getElementById('feedback');
    
    if (!videoInput.files.length) {
        feedbackDiv.innerText = "Por favor, selecciona un video.";
        return;
    }
    
    const file = videoInput.files[0];
    const formData = new FormData();
    formData.append('video', file);
    
    feedbackDiv.innerText = "Procesando video... Espere";
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            feedbackDiv.innerText = data.error;
        } else {
            feedbackDiv.innerText = data.feedback;
        }
    })
    .catch(error => {
        feedbackDiv.innerText = "Error al procesar el video: " + error;
    });
}