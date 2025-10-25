const signalEl = document.getElementById("signal");
const confEl = document.getElementById("confidence");
const closeEl = document.getElementById("latest-close");
const timeEl = document.getElementById("updated");
const refreshBtn = document.getElementById("refresh-btn");
const ctx = document.getElementById("probChart").getContext("2d");

let probChart = new Chart(ctx, {
    type: "bar",
    data: {
        labels: ["SELL", "HOLD", "BUY"],
        datasets: [{
            label: "Prediction Probability",
            data: [0, 0, 0],
            backgroundColor: ["#ff4d4d", "#ffd700", "#00ff8c"],
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true,
                max: 1,
                ticks: { color: "#fff" },
            },
            x: {
                ticks: { color: "#fff" },
            }
        },
        plugins: {
            legend: { labels: { color: "#fff" } }
        }
    }
});

async function fetchSignal() {
    signalEl.textContent = "Loading...";
    signalEl.className = "";
    try {
        const res = await fetch("/predict_vix75");
        const data = await res.json();

        if (data.error) {
            signalEl.textContent = "Error";
            confEl.textContent = "Failed to fetch signal.";
            return;
        }

        const signal = data.signal.toUpperCase();
        const conf = data.confidence;
        const probs = data.probabilities;
        const close = data.latest_close;

        signalEl.textContent = signal;
        signalEl.className = signal.toLowerCase();

        confEl.textContent = `Confidence: ${(conf * 100).toFixed(2)}%`;
        closeEl.textContent = `Latest Price: ${close.toFixed(2)}`;
        timeEl.textContent = `Updated: ${new Date(data.timestamp * 1000).toLocaleTimeString()}`;

        probChart.data.datasets[0].data = [probs.SELL, probs.HOLD, probs.BUY];
        probChart.update();
    } catch (err) {
        signalEl.textContent = "Error fetching data.";
        console.error(err);
    }
}

refreshBtn.addEventListener("click", fetchSignal);
fetchSignal();
setInterval(fetchSignal, 30000); // auto-refresh every 30s
