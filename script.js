let history = [];

document.getElementById('classifyBtn').addEventListener('click', classifyText);
document.getElementById('resetBtn').addEventListener('click', resetForm);
document.getElementById('saveBtn').addEventListener('click', saveResults);
document.getElementById('darkModeToggle').addEventListener('click', toggleDarkMode);
document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);

function classifyText() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) {
        alert('Please enter some text!');
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        displayResults(text, data.predictions);
        addToHistory(text, data.predictions);
        document.getElementById('saveBtn').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while classifying the text.');
    });
}

function displayResults(text, predictions) {
    const resultContent = document.getElementById('resultContent');
    resultContent.innerHTML = '';

    const timestamp = new Date().toUTCString();
    resultContent.innerHTML += `<p><strong>Input Text:</strong> "${text}"</p>`;
    resultContent.innerHTML += `<p><strong>Timestamp (UTC):</strong> ${timestamp}</p>`;
    resultContent.innerHTML += `<p><strong>User:</strong> kirubaharan181</p>`;
    resultContent.innerHTML += `<h3>Model Predictions:</h3>`;

    for (const [model, prediction] of Object.entries(predictions)) {
        let className = 'not-predicted';
        if (prediction === 'Hate Speech') className = 'hate';
        else if (prediction === 'Non-Hate Speech') className = 'non-hate';
        resultContent.innerHTML += `<div class="model-result"><span>${model}:</span> <span class="${className}">${prediction}</span></div>`;
    }
}

function addToHistory(text, predictions) {
    const historyContent = document.getElementById('historyContent');
    const timestamp = new Date().toUTCString();
    let historyItem = `<div class="history-item">`;
    historyItem += `<p><strong>Text:</strong> "${text}" <em>(${timestamp})</em></p>`;
    for (const [model, prediction] of Object.entries(predictions)) {
        historyItem += `<p>${model}: ${prediction}</p>`;
    }
    historyItem += `</div>`;
    history.unshift(historyItem);
    historyContent.innerHTML = history.join('');
}

function resetForm() {
    document.getElementById('inputText').value = '';
    document.getElementById('resultContent').innerHTML = '';
    document.getElementById('saveBtn').style.display = 'none';
}

function saveResults() {
    const resultContent = document.getElementById('resultContent').innerText;
    const blob = new Blob([resultContent], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `prediction_results_${new Date().toISOString().replace(/:/g, '-')}.txt`;
    a.click();
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const icon = document.querySelector('#darkModeToggle i');
    icon.classList.toggle('fa-moon');
    icon.classList.toggle('fa-sun');
}

function clearHistory() {
    history = [];
    document.getElementById('historyContent').innerHTML = '';
}