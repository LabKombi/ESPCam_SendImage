<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Driver Drowsiness Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #2a2a2a;
            color: white;
            padding: 20px;
        }

        video, canvas {
            margin: 10px auto;
            display: block;
            max-width: 100%;
        }

        #result {
            font-size: 1.2em;
            margin-top: 20px;
        }

        #toggleButton {
            padding: 10px 20px;
            font-size: 1em;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 20px;
        }

        #toggleButton:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>Driver Drowsiness Detection</h1>
    <video id="video" autoplay></video>
    <canvas id="canvas" hidden></canvas>
    <div id="result"></div>
    <button id="toggleButton">Start</button>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const resultDiv = document.getElementById('result');
        const toggleButton = document.getElementById('toggleButton');

        // const API_URL = "process.php"; // Pastikan endpoint benar
        const API_URL = "http://127.0.0.1:5000/predict"; // Pastikan endpoint benar

        let captureInterval = null;
        let isCapturing = false;

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                console.error("Error accessing the camera: ", err);
                alert("Could not access the camera. Please check permissions.");
            });

        function captureAndPredict() {
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');

                fetch(API_URL, {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Server responded with status ${response.status}`);
                    }
                    return response.json(); // Parsing respons JSON
                })
                .then(data => {
                    console.log("Data from server:", data);
                    if (data.prediction && data.confidence) {
                        resultDiv.innerHTML = `
                            <p>Prediction: ${data.prediction}</p>
                            <p>Confidence: ${data.confidence.toFixed(2)}</p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<p>Invalid response format: ${JSON.stringify(data)}</p>`;
                    }
                })
                .catch(err => {
                    console.error("Error during request: ", err);
                    resultDiv.innerHTML = `<p>Error: ${err.message}</p>`;
                });
            }, 'image/jpeg');
        }

        toggleButton.addEventListener('click', () => {
            if (isCapturing) {
                clearInterval(captureInterval);
                toggleButton.textContent = 'Start';
            } else {
                captureInterval = setInterval(captureAndPredict, 3000);
                toggleButton.textContent = 'Stop';
            }
            isCapturing = !isCapturing;
        });
    </script>
</body>
</html>
