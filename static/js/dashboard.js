/**
 * CropCare AI — Evaluation dashboard logic (frontend/admin.html).
 *
 * Responsibilities:
 *  - GET /api/evaluation from the Flask backend.
 *  - Render KPI cards, a performance table, a Chart.js category
 *    distribution chart, a 4x4 confusion matrix, and a zero-shot vs.
 *    few-shot comparison table — all from the API response.
 *  - Show "No evaluation results available yet." when the backend
 *    reports no evaluation has been run, instead of rendering any
 *    placeholder numbers or charts.
 *  - Never hardcode a metric value: every number rendered on this page
 *    comes from the fetched JSON, not from this script.
 */

(function () {
  "use strict";

  var EVALUATION_ENDPOINT = "/api/evaluation";

  // Category display order also used for the confusion matrix and chart.
  var CATEGORY_ORDER = ["Symptoms", "Prevention", "Management", "General Information"];
  var CATEGORY_COLORS = {
    "Symptoms": "#C1666B",              // --cc-rust
    "Prevention": "#52B788",            // --cc-leaf
    "Management": "#8C6239",            // --cc-soil
    "General Information": "#457B9D",   // --cc-sky
  };

  var categoryChartInstance = null;

  document.addEventListener("DOMContentLoaded", function () {
    var refreshBtn = document.getElementById("refreshBtn");
    if (!refreshBtn) return; // Not on the admin page.

    refreshBtn.addEventListener("click", loadEvaluation);
    loadEvaluation();
  });

  // ---------------------------------------------------------------------
  // Fetch + top-level state switching
  // ---------------------------------------------------------------------

  function loadEvaluation() {
    showState("loading");

    fetch(EVALUATION_ENDPOINT, { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("HTTP " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        if (!data || data.available !== true) {
          showEmptyState(data && data.message);
          return;
        }
        renderDashboard(data);
        showState("dashboard");
      })
      .catch(function (err) {
        showErrorState(err && err.message);
      });
  }

  /** Toggle between loading / empty / error / dashboard sections. */
  function showState(state) {
    var loading = document.getElementById("loadingState");
    var empty = document.getElementById("emptyState");
    var error = document.getElementById("errorState");
    var dashboard = document.getElementById("dashboardContent");

    loading.classList.toggle("d-none", state !== "loading");
    empty.classList.toggle("d-none", state !== "empty");
    error.classList.toggle("d-none", state !== "error");
    dashboard.classList.toggle("d-none", state !== "dashboard");

    setStatusPill(state);
  }

  function showEmptyState(message) {
    var messageEl = document.getElementById("emptyStateMessage");
    if (message) {
      messageEl.textContent = message;
    }
    showState("empty");
  }

  function showErrorState(message) {
    var messageEl = document.getElementById("errorStateMessage");
    if (message) {
      messageEl.textContent =
        "Check that the Flask backend is running, then try Refresh. (" + message + ")";
    }
    showState("error");
  }

  function setStatusPill(state) {
    var pill = document.getElementById("statusPill");
    var text = document.getElementById("statusPillText");
    pill.classList.remove("cc-adm-status-live", "cc-adm-status-empty");

    if (state === "dashboard") {
      pill.classList.add("cc-adm-status-live");
      text.textContent = "Live evaluation results";
      pill.querySelector(".bi").className = "bi bi-broadcast";
    } else if (state === "loading") {
      pill.classList.add("cc-adm-status-empty");
      text.textContent = "Checking for evaluation data…";
      pill.querySelector(".bi").className = "bi bi-hourglass-split";
    } else {
      pill.classList.add("cc-adm-status-empty");
      text.textContent = state === "error" ? "Couldn't reach API" : "No evaluation data yet";
      pill.querySelector(".bi").className = "bi bi-exclamation-triangle";
    }
  }

  // ---------------------------------------------------------------------
  // Rendering — driven entirely by the fetched payload
  // ---------------------------------------------------------------------

  function renderDashboard(data) {
    renderGeneratedAtNote(data);
    renderKpis(data);
    renderPerformanceTable(data.metrics);
    renderCategoryChart(data.category_distribution);
    renderConfusionMatrix(data.confusion_matrix);
    renderMethodComparison(data.method_comparison);
  }

  function renderGeneratedAtNote(data) {
    var note = document.getElementById("generatedAtNote");
    var parts = [];
    if (data.dataset && data.dataset.path) {
      parts.push("Source: " + data.dataset.path);
    }
    if (data.generated_at_utc) {
      parts.push("Generated " + formatTimestamp(data.generated_at_utc));
    }
    note.innerHTML = parts.length
      ? '<i class="bi bi-check2-circle me-1" aria-hidden="true"></i>' +
        parts.join(" &nbsp;•&nbsp; ") +
        " &nbsp;•&nbsp; actual evaluation output, not sample/demo data."
      : "";
  }

  function renderKpis(data) {
    var metrics = data.metrics || {};
    var diagnostics = data.diagnostics || {};
    var dataset = data.dataset || {};

    setText("kpiTotal", formatInt(data.total_test_questions));
    setText(
      "kpiTotalSub",
      dataset.scored_rows != null
        ? dataset.scored_rows + " scored"
        : ""
    );
    setText("kpiAccuracy", formatPercent(metrics.accuracy));
    setText("kpiPrecision", formatPercent(metrics.precision_macro));
    setText("kpiRecall", formatPercent(metrics.recall_macro));
    setText("kpiF1", formatPercent(metrics.f1_macro));
    setText("kpiScored", formatInt(dataset.scored_rows));
    setText("kpiFailures", formatInt(dataset.excluded_system_failures));
    setText("kpiUnable", formatInt(diagnostics.unable_to_classify_predictions));
  }

  function renderPerformanceTable(metrics) {
    metrics = metrics || {};
    var rows = [
      ["Accuracy", metrics.accuracy],
      ["Precision (macro)", metrics.precision_macro],
      ["Recall (macro)", metrics.recall_macro],
      ["F1 Score (macro)", metrics.f1_macro],
    ];

    var tbody = document.getElementById("performanceTableBody");
    tbody.innerHTML = "";
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var tdMetric = document.createElement("td");
      tdMetric.textContent = row[0];
      var tdScore = document.createElement("td");
      tdScore.className = "cc-adm-num";
      tdScore.textContent = formatPercent(row[1]);
      tr.appendChild(tdMetric);
      tr.appendChild(tdScore);
      tbody.appendChild(tr);
    });
  }

  function renderCategoryChart(distribution) {
    distribution = distribution || {};
    var canvas = document.getElementById("categoryChart");
    if (!canvas || typeof Chart === "undefined") return;

    var labels = CATEGORY_ORDER.filter(function (c) {
      return Object.prototype.hasOwnProperty.call(distribution, c);
    });
    var values = labels.map(function (c) {
      return distribution[c] == null ? 0 : distribution[c];
    });
    var colors = labels.map(function (c) {
      return CATEGORY_COLORS[c] || "#457B9D";
    });

    if (categoryChartInstance) {
      categoryChartInstance.destroy();
    }

    categoryChartInstance = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Test-set questions",
            data: values,
            backgroundColor: colors,
            borderRadius: 6,
            maxBarThickness: 64,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.parsed.y + " question(s)";
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { precision: 0 },
            title: { display: true, text: "Number of test questions" },
          },
        },
      },
    });
  }

  function renderConfusionMatrix(confusion) {
    confusion = confusion || {};
    var labels = confusion.labels && confusion.labels.length ? confusion.labels : CATEGORY_ORDER;
    var matrix = confusion.matrix || [];
    var table = document.getElementById("confusionTable");
    table.innerHTML = "";

    if (!matrix.length) {
      var emptyRow = document.createElement("tr");
      var emptyCell = document.createElement("td");
      emptyCell.textContent = "Confusion matrix not available yet.";
      emptyCell.colSpan = labels.length + 1;
      emptyRow.appendChild(emptyCell);
      table.appendChild(emptyRow);
      return;
    }

    // Header row.
    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    var corner = document.createElement("th");
    corner.className = "cc-adm-corner";
    corner.innerHTML = "Actual \\ Predicted";
    headRow.appendChild(corner);
    labels.forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    // Body rows.
    var tbody = document.createElement("tbody");
    labels.forEach(function (rowLabel, i) {
      var tr = document.createElement("tr");
      var rowHeader = document.createElement("td");
      rowHeader.className = "cc-adm-row-label";
      rowHeader.textContent = rowLabel;
      tr.appendChild(rowHeader);

      var rowValues = matrix[i] || [];
      labels.forEach(function (colLabel, j) {
        var td = document.createElement("td");
        var value = rowValues[j] != null ? rowValues[j] : 0;
        td.textContent = value;
        if (i === j) {
          td.classList.add("cc-adm-diag");
        } else if (value > 0) {
          td.classList.add("cc-adm-off-diag-hit");
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  function renderMethodComparison(comparison) {
    var availableEl = document.getElementById("comparisonAvailable");
    var unavailableEl = document.getElementById("comparisonUnavailable");

    if (!comparison || comparison.available !== true) {
      availableEl.classList.add("d-none");
      unavailableEl.classList.remove("d-none");
      setText(
        "comparisonUnavailableMessage",
        (comparison && comparison.message) ||
          "No zero-shot vs. few-shot comparison available yet."
      );
      return;
    }

    unavailableEl.classList.add("d-none");
    availableEl.classList.remove("d-none");

    var zeroShot = comparison.baseline_zero_shot || {};
    var fewShot = comparison.proposed_few_shot || {};
    var deltas = comparison.delta_few_shot_minus_zero_shot || {};

    var rows = [
      ["Accuracy", "accuracy"],
      ["Precision (macro)", "precision_macro"],
      ["Recall (macro)", "recall_macro"],
      ["F1 Score (macro)", "f1_macro"],
    ];

    var tbody = document.getElementById("comparisonTableBody");
    tbody.innerHTML = "";
    rows.forEach(function (row) {
      var label = row[0];
      var key = row[1];
      var tr = document.createElement("tr");

      var tdLabel = document.createElement("td");
      tdLabel.textContent = label;
      tr.appendChild(tdLabel);

      var tdZero = document.createElement("td");
      tdZero.className = "cc-adm-num";
      tdZero.textContent = formatPercent(zeroShot[key]);
      tr.appendChild(tdZero);

      var tdFew = document.createElement("td");
      tdFew.className = "cc-adm-num";
      tdFew.textContent = formatPercent(fewShot[key]);
      tr.appendChild(tdFew);

      var tdDelta = document.createElement("td");
      tdDelta.className = "cc-adm-num";
      var deltaValue = deltas[key];
      tdDelta.textContent = formatSignedPercent(deltaValue);
      if (typeof deltaValue === "number") {
        tdDelta.classList.add(
          deltaValue > 0
            ? "cc-adm-comparison-delta-up"
            : deltaValue < 0
            ? "cc-adm-comparison-delta-down"
            : "cc-adm-comparison-delta-flat"
        );
      }
      tr.appendChild(tdDelta);

      tbody.appendChild(tr);
    });

    var conclusionEl = document.getElementById("comparisonConclusion");
    if (comparison.conclusion) {
      conclusionEl.textContent = comparison.conclusion;
      conclusionEl.classList.remove("d-none");
    } else {
      conclusionEl.classList.add("d-none");
    }
  }

  // ---------------------------------------------------------------------
  // Formatting helpers
  // ---------------------------------------------------------------------

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function formatInt(value) {
    if (value === null || value === undefined || isNaN(value)) return "—";
    return String(value);
  }

  function formatPercent(value) {
    if (typeof value !== "number" || isNaN(value)) return "—";
    return (value * 100).toFixed(1) + "%";
  }

  function formatSignedPercent(value) {
    if (typeof value !== "number" || isNaN(value)) return "—";
    var sign = value > 0 ? "+" : "";
    return sign + (value * 100).toFixed(1) + " pts";
  }

  function formatTimestamp(isoString) {
    var date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
})();
