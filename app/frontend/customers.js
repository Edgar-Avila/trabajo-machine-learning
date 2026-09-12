const tbody = document.querySelector("#customers");
const summary = document.querySelector("#summary");
const pagination = document.querySelector("#pagination");
const dashboard = document.querySelector("#dashboard");
const modal = document.querySelector("#customer-modal");
const modalTitle = document.querySelector("#modal-title");
const modalContent = document.querySelector("#modal-content");
const PAGE_SIZE = 10;
let currentPage = 1;

async function load(page) {
  const response = await fetch(`/customers?page=${page}&page_size=${PAGE_SIZE}`);
  if (!response.ok) {
    summary.textContent = "No se pudieron cargar los clientes.";
    return;
  }
  const data = await response.json();
  tbody.innerHTML = "";
  for (const row of data.items) {
    tbody.insertAdjacentHTML(
      "beforeend",
      "<tr><td>" +
        escapeHtml(row.CustomerID) +
        "</td><td>" +
        row.Age +
        "</td><td>" +
        escapeHtml(row.Gender) +
        "</td><td>" +
        escapeHtml(row["Subscription Type"]) +
        "</td><td>" +
        escapeHtml(row["Contract Length"]) +
        "</td><td>" +
        (row.ChurnProbability * 100).toFixed(1) +
        "%</td><td>" +
        escapeHtml(row.RiskLevel) +
        "</td><td>v" +
        row.ModelVersion +
        '</td><td><button class="eye" data-id="' +
        escapeHtml(row.CustomerID) +
        '" aria-label="Ver historial">👁</button></td></tr>',
    );
  }
  currentPage = page;
  summary.textContent =
    "Página " + data.page + " de " + data.total_pages + " · " + data.total + " clientes · modelo v" +
    (data.items[0] ? data.items[0].ModelVersion : "-");
  renderPagination(data);
}

tbody.addEventListener("click", async (event) => {
  const button = event.target.closest("button.eye");
  if (!button) return;
  await openHistory(button.dataset.id);
});

function renderPagination(data) {
  pagination.innerHTML = "";
  if (data.total_pages <= 1) return;
  const prev = document.createElement("button");
  prev.textContent = "‹ Anterior";
  prev.disabled = data.page <= 1;
  prev.addEventListener("click", () => load(data.page - 1));
  pagination.appendChild(prev);
  const next = document.createElement("button");
  next.textContent = "Siguiente ›";
  next.disabled = data.page >= data.total_pages;
  next.addEventListener("click", () => load(data.page + 1));
  pagination.appendChild(next);
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

async function openHistory(customerId) {
  const response = await fetch(`/customers/${encodeURIComponent(customerId)}/history`);
  if (!response.ok) {
    modalTitle.textContent = "Cliente " + customerId;
    modalContent.innerHTML = "<p>No se pudo cargar el historial.</p>";
    showModal();
    return;
  }
  const data = await response.json();
  modalTitle.textContent =
    "Historial de churn — cliente " + data.customer_id + (data.churn_label != null ? " · Churn real: " + data.churn_label : "");
  const lineConfig = {
    type: "line",
    data: {
      labels: data.history.map((h) => "v" + h.version),
      datasets: [
        {
          label: "Churn",
          data: data.history.map((h) => +(h.churn_probability * 100).toFixed(1)),
          borderColor: "#1657b7",
          backgroundColor: "rgba(22,87,183,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 4,
        },
      ],
    },
    options: {
      scales: { y: { ticks: { callback: (value) => value + "%" } } },
      plugins: {
        legend: { display: false },
        datalabels: { align: "top", color: "#172033", font: { weight: "bold" } },
      },
    },
  };
  modalContent.innerHTML =
    '<img class="history-chart" alt="Churn por versión" src="' +
    chartUrl(lineConfig, 500, 260) +
    '"><table><thead><tr><th>Versión</th><th>Churn</th><th>Riesgo</th><th>Δ</th><th>Predicho el</th></tr></thead><tbody>' +
    data.history
      .map(
        (h) =>
          "<tr><td>v" +
          h.version +
          "</td><td>" +
          (h.churn_probability * 100).toFixed(1) +
          "%</td><td>" +
          escapeHtml(h.risk_level) +
          "</td><td>" +
          (h.delta == null
            ? "—"
            : (h.delta >= 0 ? "▲ +" : "▼ ") + (h.delta * 100).toFixed(1) + "%") +
          "</td><td>" +
          escapeHtml(h.predicted_at.slice(0, 19).replace("T", " ")) +
          "</td></tr>",
      )
      .join("") +
    "</tbody></table>";
  showModal();
}

function showModal() {
  modal.classList.remove("hidden");
  document.querySelector("#modal-close").focus();
}

function closeModal() {
  modal.classList.add("hidden");
}

document.querySelector("#modal-close").addEventListener("click", closeModal);
modal.addEventListener("click", (event) => {
  if (event.target === modal) closeModal();
});

async function loadSummary() {
  const response = await fetch("/customers/summary");
  if (!response.ok) return;
  const data = await response.json();
  const total = data.total_customers;
  const cards = [
    ["Total clientes", total],
    ["Churn medio", (data.avg_churn_probability * 100).toFixed(1) + "%"],
  ];
  for (const risk of data.risk_counts) {
    cards.push([risk.level, risk.count + " (" + risk.percentage + "%)"]);
  }
  const riskConfig = {
    type: "doughnut",
    data: {
      labels: data.risk_counts.map((r) => r.level),
      datasets: [
        {
          data: data.risk_counts.map((r) => r.count),
          backgroundColor: ["#22a06b", "#e6a23c", "#d9534f"],
        },
      ],
    },
    options: {
      plugins: {
        legend: { position: "bottom" },
        datalabels: {
          color: "#ffffff",
          font: { weight: "bold" },
          textShadowColor: "rgba(0,0,0,0.4)",
          textShadowBlur: 4,
        },
      },
    },
  };
  dashboard.innerHTML =
    '<div class="cards">' +
    cards
      .map(
        (c) =>
          '<div class="card"><span class="card-label">' +
          escapeHtml(c[0]) +
          '</span><span class="card-value">' +
          c[1] +
          "</span></div>",
      )
      .join("") +
    "</div><div class=\"charts-row\">" +
    '<figure class="chart"><img alt="Churn por suscripción" src="' +
    chartUrl(barConfig("Churn medio por suscripción", data.by_subscription), 380, 280) +
    '"><figcaption>Churn medio por suscripción</figcaption></figure>' +
    '<figure class="chart"><img alt="Churn por contrato" src="' +
    chartUrl(barConfig("Churn medio por contrato", data.by_contract), 380, 280) +
    '"><figcaption>Churn medio por contrato</figcaption></figure>' +
    "</div><div class=\"charts-row\">" +
    '<figure class="chart"><img alt="Distribución de riesgo" src="' +
    chartUrl(riskConfig, 380, 300) +
    '"><figcaption>Clientes por nivel de riesgo</figcaption></figure>' +
    "</div>";
}

function chartUrl(config, width, height) {
  const params = new URLSearchParams({
    chart: JSON.stringify(config),
    width: String(width),
    height: String(height),
    version: "4",
  });
  return "https://quickchart.io/chart?" + params.toString();
}

function barConfig(label, groups) {
  return {
    type: "bar",
    data: {
      labels: groups.map((g) => g.name),
      datasets: [
        {
          label: label,
          data: groups.map((g) => +(g.avg_churn_probability * 100).toFixed(1)),
          backgroundColor: "#1657b7",
        },
      ],
    },
    options: {
      scales: { y: { ticks: { callback: (value) => value + "%" } } },
    },
  };
}

load(1);
loadSummary();