document.addEventListener('DOMContentLoaded', () => {
    const tickerInput = document.getElementById('tickerInput');
    const predictBtn = document.getElementById('predictBtn');
    const loader = document.getElementById('loader');
    const metricsContainer = document.getElementById('metricsContainer');
    const latestPrice = document.getElementById('latestPrice');
    const predictedPrice = document.getElementById('predictedPrice');
    const chartContainer = document.getElementById('chartContainer');
    const initialMessage = document.getElementById('initialMessage'); // Get the initial message element

    const handlePrediction = async () => {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (!ticker) {
            chartContainer.innerHTML = '<p class="error-message">Please enter a stock ticker.</p>';
            initialMessage.style.display = 'none';
            return;
        }

        loader.style.display = 'block';
        metricsContainer.style.display = 'none';
        chartContainer.innerHTML = '';
        initialMessage.style.display = 'none'; // Hide the initial message
        latestPrice.textContent = '$-';
        predictedPrice.textContent = '$-';

        try {
            // 1. Fetch historical data
            const historyResponse = await fetch(`/api/history/${ticker}?period=1y`);
            if (!historyResponse.ok) {
                const errorData = await historyResponse.json();
                throw new Error(errorData.detail || `Failed to fetch data. Status: ${historyResponse.status}`);
            }
            const historyData = await historyResponse.json();

            if (!historyData || historyData.length < 60) {
                throw new Error('Insufficient historical data for prediction (need at least 60 days).');
            }

            // 2. Prepare data for prediction and display
            const recentData = historyData.slice(-60);
            const closes = recentData.map(item => item.Close);
            const latestClose = closes[closes.length - 1];
            latestPrice.textContent = `$${latestClose.toFixed(2)}`;

            // 3. Get prediction from the backend
            const predictResponse = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: closes })
            });

            if (!predictResponse.ok) {
                throw new Error(`Prediction failed. Status: ${predictResponse.status}`);
            }
            const prediction = await predictResponse.json();
            predictedPrice.textContent = `$${prediction.prediction.toFixed(2)}`;

            // 4. Plot the results
            const dates = recentData.map(item => item.Date);
            const lastDate = new Date(dates[dates.length - 1]);
            lastDate.setDate(lastDate.getDate() + 2);
            const nextDay = lastDate.toISOString().split('T')[0];

            const traceHistorical = {
                x: dates,
                y: closes,
                mode: 'lines',
                name: 'Historical Price',
                line: { color: '#007bff' }
            };

            const tracePredicted = {
                x: [dates[dates.length - 1], nextDay],
                y: [latestClose, prediction.prediction],
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: { color: '#ff6347', dash: 'dash' },
                marker: { color: '#ff6347', size: 8 }
            };

            const layout = {
                title: `${ticker} Stock Price Prediction`,
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price (USD)' },
                paper_bgcolor: '#f9f9f9',
                plot_bgcolor: '#f9f9f9',
                showlegend: true
            };

            Plotly.newPlot(chartContainer, [traceHistorical, tracePredicted], layout);

            metricsContainer.style.display = 'flex';

        } catch (error) {
            console.error('Error:', error);
            chartContainer.innerHTML = `<p class="error-message">⚠️ ${error.message}</p>`;
        } finally {
            loader.style.display = 'none';
        }
    };

    predictBtn.addEventListener('click', handlePrediction);
    tickerInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            handlePrediction();
        }
    });

    // ✅ REMOVED: The line below was causing the prediction to run on page load.
    // handlePrediction(); 
});