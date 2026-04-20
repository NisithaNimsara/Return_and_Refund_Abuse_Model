const API_BASE_URL = "http://localhost:8000";
    const SIGNAL_META = [
      { key: "changed_mind", label: "Changed Mind Return", icon: "💭" },
      { key: "high_discount", label: "High Discount + Return", icon: "🏷️" },
      { key: "late_return", label: "Late Return (>90 days)", icon: "⏰" },
      { key: "bulk_return", label: "Bulk Quantity Return", icon: "📦" },
      { key: "expensive_item", label: "High-Value Item Return", icon: "💰" }
    ];

    const exampleData = {
      Product_Price: 420,
      Discount_Applied: 42,
      Order_Quantity: 2,
      Days_to_Return: 20,
      Return_Status: "Returned",
      Return_Reason: "Changed mind",
      User_Age: 29
    };

    const form = document.getElementById("prediction-form");
    const submitBtn = document.getElementById("submit-btn");
    const fillBtn = document.getElementById("fill-btn");
    const resetBtn = document.getElementById("reset-btn");
    const statusEl = document.getElementById("status");
    const emptyStateEl = document.getElementById("empty-state");
    const summaryEl = document.getElementById("summary");
    const verdictBoxEl = document.getElementById("verdict-box");
    const verdictTextEl = document.getElementById("verdict-text");
    const verdictSubtextEl = document.getElementById("verdict-subtext");
    const probabilityChipEl = document.getElementById("probability-chip");
    const statVerdictEl = document.getElementById("stat-verdict");
    const statScoreEl = document.getElementById("stat-score");
    const signalsEl = document.getElementById("signals");
    const reasoningEl = document.getElementById("reasoning");

    function setStatus(type, message) {
      if (!message) {
        statusEl.className = "status";
        statusEl.textContent = "";
        return;
      }
      statusEl.className = `status show ${type}`;
      statusEl.textContent = message;
    }

    function setLoading(isLoading) {
      submitBtn.disabled = isLoading;
      submitBtn.textContent = isLoading ? "Analyzing..." : "Analyze Return";
    }

    function hideResults() {
      summaryEl.classList.remove("show");
      emptyStateEl.style.display = "block";
    }

    function showResults() {
      summaryEl.classList.add("show");
      emptyStateEl.style.display = "none";
    }

    function buildSummaryText(result) {
      if (result.verdict === "ABUSE") {
        return `The backend flagged this return for escalation with an abuse probability of ${result.probability}. Review the triggered signals and reasoning below.`;
      }
      return `The backend classified this case as not abusive with an abuse probability of ${result.probability}. The signal breakdown and reasoning are shown below.`;
    }

    function renderSignals(signals) {
      signalsEl.innerHTML = "";
      SIGNAL_META.forEach((item) => {
        const active = Boolean(signals?.[item.key]);
        const row = document.createElement("div");
        row.className = "signal-item";
        row.innerHTML = `
          <div class="signal-left">
            <div class="signal-icon">${item.icon}</div>
            <div class="signal-name">${item.label}</div>
          </div>
          <div class="signal-state ${active ? "on" : "off"}">${active ? "Triggered" : "Not Triggered"}</div>
        `;
        signalsEl.appendChild(row);
      });
    }

    function renderResult(result) {
      const isAbuse = result.verdict === "ABUSE";
      verdictBoxEl.className = `verdict ${isAbuse ? "abuse" : "safe"}`;
      verdictTextEl.textContent = result.verdict || "-";
      probabilityChipEl.textContent = result.probability || "-";
      verdictSubtextEl.textContent = buildSummaryText(result);
      statVerdictEl.textContent = result.verdict || "-";
      statScoreEl.textContent = `${result.abuse_score ?? "-"}/5`;
      reasoningEl.textContent = result.reasoning || "No reasoning returned by the API.";
      renderSignals(result.signals || {});
      showResults();
    }

    function getPayload() {
      const data = new FormData(form);
      return {
        Product_Price: Number(data.get("Product_Price")),
        Discount_Applied: Number(data.get("Discount_Applied")),
        Order_Quantity: Number(data.get("Order_Quantity")),
        Days_to_Return: Number(data.get("Days_to_Return")),
        Return_Status: String(data.get("Return_Status")),
        Return_Reason: String(data.get("Return_Reason")),
        User_Age: Number(data.get("User_Age"))
      };
    }

    function fillExample() {
      Object.entries(exampleData).forEach(([key, value]) => {
        const field = form.elements.namedItem(key);
        if (field) field.value = value;
      });
    }

    async function submitPrediction(event) {
      event.preventDefault();
      setStatus("loading", `Calling ${API_BASE_URL}/predict ...`);
      setLoading(true);

      try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(getPayload())
        });

        const contentType = response.headers.get("content-type") || "";
        const body = contentType.includes("application/json") ? await response.json() : await response.text();

        if (!response.ok) {
          const detail = typeof body === "object" && body?.detail
            ? body.detail
            : typeof body === "string"
              ? body
              : "The API request failed.";
          throw new Error(detail);
        }

        renderResult(body);
        setStatus("", "");
      } catch (error) {
        hideResults();
        setStatus(
          "error",
          `Could not get a response from the backend API at ${API_BASE_URL}/predict. ${error?.message || "Please make sure the server is running."}`
        );
      } finally {
        setLoading(false);
      }
    }

    form.addEventListener("submit", submitPrediction);
    fillBtn.addEventListener("click", fillExample);
    resetBtn.addEventListener("click", () => {
      setStatus("", "");
      hideResults();
    });
