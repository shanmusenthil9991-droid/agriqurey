/**
 * CropCare AI — History page logic.
 *
 * Reads every classification record saved by static/js/classifier.js
 * (localStorage key "cropcare:history", an array with the newest
 * question first) and renders the full list, or an empty state if
 * nothing has been asked yet on this device.
 */

(function () {
  "use strict";

  var HISTORY_STORAGE_KEY = "cropcare:history";
  var FALLBACK_CATEGORY_NORMALIZED = "unable to classify";

  // Purely presentational: icon + accent per known category label.
  var CATEGORY_PRESENTATION = {
    "symptoms": { icon: "🩺", slug: "symptoms" },
    "prevention": { icon: "🛡️", slug: "prevention" },
    "management": { icon: "🧰", slug: "management" },
    "general information": { icon: "📘", slug: "general-information" },
    "unable to classify": { icon: "❓", slug: "unable-to-classify" }
  };
  var DEFAULT_PRESENTATION = { icon: "🌱", slug: "other" };

  document.addEventListener("DOMContentLoaded", function () {
    var listEl = document.getElementById("historyList");
    var actionsEl = document.getElementById("historyActions");
    var emptyEl = document.getElementById("historyEmpty");
    var clearBtn = document.getElementById("clearHistoryBtn");

    // Not on the history page (defensive — file is shared per-project).
    if (!listEl || !emptyEl) return;

    render();

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        if (!window.confirm("Clear your entire question history on this device?")) return;
        try {
          localStorage.removeItem(HISTORY_STORAGE_KEY);
        } catch (err) {
          // Ignore — storage may be unavailable; render() will reflect reality.
        }
        render();
      });
    }

    function render() {
      var history = readHistory();

      if (!history.length) {
        listEl.classList.add("d-none");
        if (actionsEl) actionsEl.style.display = "none";
        emptyEl.classList.remove("d-none");
        return;
      }

      emptyEl.classList.add("d-none");
      listEl.classList.remove("d-none");
      if (actionsEl) actionsEl.style.display = "";
      listEl.innerHTML = "";
      history.forEach(function (record) {
        listEl.appendChild(buildHistoryCard(record));
      });
    }

    function readHistory() {
      var raw;
      try {
        raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      } catch (err) {
        return [];
      }
      if (!raw) return [];
      try {
        var parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.filter(isValidRecord) : [];
      } catch (err) {
        return [];
      }
    }

    function isValidRecord(record) {
      return (
        record &&
        typeof record === "object" &&
        typeof record.category === "string" &&
        record.category.trim() !== "" &&
        typeof record.confidence === "number" &&
        !isNaN(record.confidence) &&
        typeof record.reason === "string"
      );
    }

    function buildHistoryCard(record) {
      var normalized = record.category.trim().toLowerCase();
      var isUnclassified = normalized === FALLBACK_CATEGORY_NORMALIZED;
      var presentation = CATEGORY_PRESENTATION[normalized] || DEFAULT_PRESENTATION;

      var card = document.createElement("article");
      card.className = "cc-result-card cc-history-item" + (isUnclassified ? " cc-result-card-muted" : "");

      var head = document.createElement("div");
      head.className = "cc-result-card-head";

      var questionWrap = document.createElement("div");
      var label = document.createElement("p");
      label.className = "cc-result-label";
      label.textContent = "Question";
      var question = document.createElement("p");
      question.className = "cc-result-question";
      question.textContent = record.question || "—";
      questionWrap.appendChild(label);
      questionWrap.appendChild(question);
      head.appendChild(questionWrap);

      if (record.crop && String(record.crop).trim()) {
        var cropWrap = document.createElement("div");
        cropWrap.className = "cc-result-crop";
        cropWrap.innerHTML = '<i class="bi bi-flower1" aria-hidden="true"></i><span></span>';
        cropWrap.querySelector("span").textContent = record.crop;
        head.appendChild(cropWrap);
      }

      card.appendChild(head);
      card.appendChild(document.createElement("hr")).className = "cc-result-divider";

      var categoryRow = document.createElement("div");
      categoryRow.className = "cc-result-category-row cc-history-category-row";

      var badge = document.createElement("div");
      badge.className = "cc-category-badge";
      badge.setAttribute("data-category", presentation.slug);
      badge.innerHTML =
        '<span class="cc-category-badge-icon" aria-hidden="true">' + presentation.icon + '</span>' +
        '<span class="cc-category-badge-text"></span>';
      badge.querySelector(".cc-category-badge-text").textContent = record.category;
      categoryRow.appendChild(badge);

      if (!isUnclassified) {
        var confidence = document.createElement("span");
        confidence.className = "cc-history-confidence";
        confidence.textContent = toConfidencePercent(record.confidence) + "% confidence";
        categoryRow.appendChild(confidence);
      }

      card.appendChild(categoryRow);

      if (record.reason && record.reason.trim()) {
        var reasonBlock = document.createElement("div");
        reasonBlock.className = "cc-reason-block cc-history-reason";
        var reasonText = document.createElement("p");
        reasonText.className = "cc-reason-text";
        reasonText.textContent = record.reason;
        reasonBlock.appendChild(reasonText);
        card.appendChild(reasonBlock);
      }

      var timestampEl = document.createElement("p");
      timestampEl.className = "cc-history-timestamp";
      timestampEl.textContent = formatTimestamp(record.timestamp);
      card.appendChild(timestampEl);

      return card;
    }

    function toConfidencePercent(confidence) {
      var value = confidence <= 1 ? confidence * 100 : confidence;
      value = Math.max(0, Math.min(100, value));
      return Math.round(value);
    }

    function formatTimestamp(isoString) {
      if (!isoString) return "";
      var date = new Date(isoString);
      if (isNaN(date.getTime())) return "";
      try {
        return date.toLocaleString(undefined, {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit"
        });
      } catch (err) {
        return date.toISOString();
      }
    }
  });
})();
