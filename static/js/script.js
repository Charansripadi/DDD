// static/script.js
const video = document.getElementById('video');
const statusText = document.getElementById('status');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        setInterval(() => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');

            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === "Error") {
                    statusText.innerText = "Error loading model";
                    statusText.style.color = "red";
                } else {
                    statusText.innerText = `Status: ${data.status} | EAR: ${data.ear} | CNN Score: ${data.cnn_score}`;
                    statusText.style.color = data.status === "Drowsy" ? "red" : "lime";
                }
            });
        }, 1000);
    })
    .catch((err) => {
        statusText.innerText = "Unable to access webcam";
        statusText.style.color = "red";
        console.error("Error accessing camera: ", err);
    });
