function predict() {
    const file = document.getElementById('audioFile').files[0];
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = `Error: ${data.error}`;
        } else {
            document.getElementById('result').innerHTML = `
                Result: ${data.result}<br>
                Confidence: ${(data.confidence * 100).toFixed(2)}%
            `;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'An error occurred during prediction.';
    });
}