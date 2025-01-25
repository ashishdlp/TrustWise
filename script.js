document.addEventListener('DOMContentLoaded', () => {
    const scoreButton = document.getElementById('scoreButton');
    const textInput = document.getElementById('textInput');
    const scoresDisplay = document.getElementById('scoresDisplay');
    const historyTableBody = document.getElementById('historyTable').querySelector('tbody');
    const scoreGraph = document.getElementById('scoreGraph');

    scoreButton.addEventListener('click', () => {
        const text = textInput.value;
        if (text.trim() !== '') {
            fetch('/score_text/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json' // Explicitly accept JSON
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => {
                if (!response.ok) { // Check for HTTP errors
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                displayScores(data.scores);
                updateHistoryTable(data);
                // updateScoreGraph(data.scores);
            })
            .catch(error => {
                console.error('Fetch error:', error); // More specific error logging
                scoresDisplay.textContent = 'Error scoring text.';
            });
        } else {
            scoresDisplay.textContent = 'Please enter text to score.';
        }
    });

    function displayScores(scores) {
        scoresDisplay.textContent = JSON.stringify(scores, null, 2);
    }

    function updateHistoryTable(data) {
        const newRow = historyTableBody.insertRow(0); // Add to the top of the table
        newRow.insertCell(0).textContent = new Date().toLocaleString();
        newRow.insertCell(1).textContent = data.received_text;
        newRow.insertCell(2).textContent = JSON.stringify(data.scores.Gibberish);
        newRow.insertCell(3).textContent = JSON.stringify(data.scores.Education);
    }

    // Placeholder function for graph update - to be implemented with a charting library
    function updateScoreGraph(scores) {
        scoreGraph.textContent = 'Graph data will be displayed here.';
        // Example of how you might use scores to update a graph (using a hypothetical library)
        // For example, if you decide to use Chart.js, you would process 'scores' to fit chart data format
        // and then update the chart.
    }

    // Initial history load - will implement fetching from database later
    function loadHistory() {
        // Placeholder for loading history from database and displaying in table and graph
        console.log("Loading history...");
    }

    loadHistory(); // Call on page load
});
