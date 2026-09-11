const form = document.querySelector("#prediction-form");
const message = document.querySelector("#message");
const rows = document.querySelector("#customers");
const numberFields = new Set([
  "Age",
  "Tenure",
  "Usage Frequency",
  "Support Calls",
  "Payment Delay",
  "Total Spend",
  "Last Interaction",
]);
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  for (const field of numberFields) data[field] = Number(data[field]);
  message.textContent = "Calculating prediction…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Prediction failed");
    rows.insertAdjacentHTML(
      "afterbegin",
      "<tr><td>" +
        escapeHtml(result.customer_id) +
        "</td><td>" +
        (result.churn_probability * 100).toFixed(1) +
        "%</td><td>" +
        result.risk_level +
        "</td></tr>",
    );
    message.textContent = "Prediction added to the list.";
  } catch (error) {
    message.textContent = error.message;
  }
});
function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}
