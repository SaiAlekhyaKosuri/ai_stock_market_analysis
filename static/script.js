document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predictBtn');
    const tickerInput = document.getElementById('tickerInput');
    const loader = document.getElementById('loader');
    
    const chartContainer = document.getElementById('chartContainer');
    const metricsContainer = document.getElementById('metricsContainer');
    const latestPriceEl = document.getElementById('latestPrice');
    const predictedPriceEl = document.getElementById('predictedPrice');

    const handlePrediction = async () => {
        const ticker = tickerInput.value.toUpperCase();
        if (!ticker) {
            alert('Please enter a stock ticker.');
            return;
        }

        // Show loader and hide old results
        loader.style.display = 'block';
        chartContainer.innerHTML = '';
        metricsContainer.style.display = 'none';

        try {
            // 1. Fetch historical data for the chart
            const historyResponse = await fetch(`/history/${ticker}`);
            if (!historyResponse.ok) throw new Error('Failed to fetch historical data.');
            
            const historyData = await historyResponse.json();
            const dates = Object.keys(historyData);
            const closePrices = dates.map(date => historyData[date].Close);

            // 2. Draw the chart using Plotly.js
            const trace = {
                x: dates,
                y: closePrices,
                type: 'scatter',
                mode: 'lines',
                name: `${ticker} Close Price`,
                line: { color: '#4FD1C5' }
            };
            const layout = {
                title: `${ticker} Closing Price History`,
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price (USD)' },
                paper_bgcolor: '#FFFFFF',
                plot_bgcolor: '#FFFFFF'
            };
            Plotly.newPlot('chartContainer', [trace], layout);

            // 3. Prepare data and call the prediction API
            const last60Days = closePrices.slice(-60);
            const predictResponse = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: last60Days })
            });
            if (!predictResponse.ok) throw new Error('Failed to get prediction.');
            
            const predictionData = await predictResponse.json();
            const latestPrice = last60Days[last60Days.length - 1];
            const predictedPrice = predictionData.prediction;

            // 4. Update the metrics
            latestPriceEl.textContent = `$${latestPrice.toFixed(2)}`;
            predictedPriceEl.textContent = `$${predictedPrice.toFixed(2)}`;
            metricsContainer.style.display = 'flex';

        } catch (error) {
            console.error('Error:', error);
            chartContainer.innerHTML = `<p style="color: red; text-align: center;">${error.message}</p>`;
        } finally {
            // Hide loader
            loader.style.display = 'none';
        }
    };

    predictBtn.addEventListener('click', handlePrediction);
    // Optional: Allow prediction on pressing Enter in the input box
    tickerInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            handlePrediction();
        }
    });
});