// static/app.js
const endpoint = "/predict_vix75";
const refreshBtn = document.getElementById("refreshBtn");
const signalText = document.getElementById("signalText");
const confidenceEl = document.getElementById("confidence");
const priceEl = document.getElementById("price");
const modelVerEl = document.getElementById("modelVer");
const lastUpdatedEl = document.getElementById("lastUpdated");
const historyList = document.getElementById("historyList");

let history = [];
let chart;

async function fetchSignal() {
  try {
    const resp = await fetch(endpoint);
    const data = await resp.json();
    if (data.error) {
      console.error("API error:", data);
      alert("API error: " + (data.detail || JSON.stringify(data.error)));
      return;
    }

    // Update UI
    signalText.textContent = data.signal;
    priceEl.textContent = `Price: ${data.latest_close.toFixed(2)}`;
    confidenceEl.textContent = `Confidence: ${(data.confidence*100).toFixed(1)}%`;
    modelVerEl.textContent = `Model: ${data.model_version}`;
    lastUpdatedEl.textContent = new Date(data.timestamp * 1000).toLocaleString();

    // color
    if (data.signal === "BUY") {
      signalText.className = "signal-text buy";
    } else if (data.signal === "SELL") {
      signalText.className = "signal-text sell";
    } else {
      signalText.className = "signal-text hold";
    }

    // update history
    history.unshift({
      timestamp: data.timestamp,
      signal: data.signal,
      confidence: data.confidence,
      price: data.latest_close
    });
    if (history.length > 30) history.pop();
    renderHistory();

    // update chart: append last price to chart
    if (chart) {
      chart.data.labels.push(new Date(data.timestamp*1000).toLocaleTimeString());
      chart.data.datasets[0].data.push(data.latest_close);
      if (chart.data.labels.length > 60) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
      }
      chart.update();
    }
  } catch (err) {
    console.error(err);
    alert("Failed to fetch signal: " + err.message);
  }
}

function renderHistory() {
  historyList.innerHTML = "";
  for (const h of history) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${h.signal}</strong> @ ${h.price.toFixed(2)} — ${(h.confidence*100).toFixed(1)}% <span class="ts">${new Date(h.timestamp*1000).toLocaleString()}</span>`;
    historyList.appendChild(li);
  }
}

function initChart() {
  const ctx = document.getElementById("priceChart").getContext("2d");
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'VIX75 Close',
        data: [],
        borderWidth: 2,
        fill: false,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { display: true },
        y: { display: true }
      }
    }
  });
}

refreshBtn.addEventListener("click", fetchSignal);

window.addEventListener("load", () => {
  initChart();
  fetchSignal();
  // auto refresh every 1 minute by default
  setInterval(fetchSignal, 60 * 1000);
});
