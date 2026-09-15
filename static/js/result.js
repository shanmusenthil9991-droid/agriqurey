/**
 * CropCare AI — Result page logic.
 *
 * Reads the classification record written by static/js/classifier.js
 * (localStorage key "cropcare:lastClassification") and renders it.
 *
 * IMPORTANT: every displayed value — category, confidence, reason,
 * question, crop — comes from that stored backend response. Nothing
 * here hardcodes "Symptoms" / "Prevention" / "Management" /
 * "General Information" as the result; those strings only ever appear
 * as icon/color lookup keys used to decorate whatever category the
 * backend actually returned.
 */

(function () {
  "use strict";

  var RESULT_STORAGE_KEY = "cropcare:lastClassification";
  var FALLBACK_CATEGORY_NORMALIZED = "unable to classify";

  // Purely presentational: icon + accent per known category label.
  // An unrecognized category still renders correctly with the default icon.
  var CATEGORY_PRESENTATION = {
    "symptoms": { icon: "🩺", slug: "symptoms" },
    "prevention": { icon: "🛡️", slug: "prevention" },
    "management": { icon: "🧰", slug: "management" },
    "general information": { icon: "📘", slug: "general-information" },
    "unable to classify": { icon: "❓", slug: "unable-to-classify" }
  };
  var DEFAULT_PRESENTATION = { icon: "🌱", slug: "other" };

  document.addEventListener("DOMContentLoaded", function () {
    var resultView = document.getElementById("resultView");
    var unclassifiedView = document.getElementById("unclassifiedView");
    var emptyView = document.getElementById("emptyView");

    // Not on the result page (defensive — file is shared per-project).
    if (!resultView || !unclassifiedView || !emptyView) return;

    var record = readStoredRecord();

    if (!record) {
      showView(emptyView);
      return;
    }

    var normalizedCategory = String(record.category || "").trim().toLowerCase();

    if (normalizedCategory === FALLBACK_CATEGORY_NORMALIZED) {
      renderUnclassified(record);
      showView(unclassifiedView);
      return;
    }

    renderResult(record);
    showView(resultView);
  });

  function showView(viewEl) {
    [document.getElementById("resultView"),
     document.getElementById("unclassifiedView"),
     document.getElementById("emptyView")].forEach(function (el) {
      if (el) el.classList.add("d-none");
    });
    viewEl.classList.remove("d-none");
  }

  /**
   * Reads and validates the stored record. Returns null if nothing is
   * stored, storage can't be read, the JSON is malformed, or required
   * fields are missing/invalid — in every one of those cases the page
   * falls back to the "no result" empty state rather than guessing.
   */
  function readStoredRecord() {
    var raw;
    try {
      raw = localStorage.getItem(RESULT_STORAGE_KEY);
    } catch (err) {
      return null; // Storage unavailable (private browsing, disabled, etc.)
    }

    if (!raw) return null;

    var parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      return null; // Corrupted JSON.
    }

    if (!parsed || typeof parsed !== "object") return null;
    if (typeof parsed.category !== "string" || !parsed.category.trim()) return null;
    if (typeof parsed.confidence !== "number" || isNaN(parsed.confidence)) return null;
    if (typeof parsed.reason !== "string") return null;

    return parsed;
  }

  function renderResult(record) {
    var normalized = record.category.trim().toLowerCase();
    var presentation = CATEGORY_PRESENTATION[normalized] || DEFAULT_PRESENTATION;

    document.getElementById("resultQuestion").textContent = record.question || "—";

    var cropWrap = document.getElementById("resultCropWrap");
    var cropEl = document.getElementById("resultCrop");
    if (record.crop && String(record.crop).trim()) {
      cropEl.textContent = record.crop;
      cropWrap.classList.remove("d-none");
    } else {
      cropWrap.classList.add("d-none");
    }

    var badge = document.getElementById("categoryBadge");
    badge.setAttribute("data-category", presentation.slug);
    document.getElementById("categoryIcon").textContent = presentation.icon;
    document.getElementById("categoryName").textContent = record.category;

    var confidencePercent = toConfidencePercent(record.confidence);
    document.getElementById("confidenceValue").textContent = confidencePercent + "%";

    var bar = document.getElementById("confidenceBar");
    bar.setAttribute("aria-valuenow", String(confidencePercent));
    var fill = document.getElementById("confidenceBarFill");
    // Set width on the next frame so the transition animates in on load.
    requestAnimationFrame(function () {
      fill.style.width = confidencePercent + "%";
    });

    document.getElementById("resultReason").textContent =
      record.reason && record.reason.trim() ? record.reason : "No reason was provided.";
  }

  function renderUnclassified(record) {
    var wrap = document.getElementById("unclassifiedQuestionWrap");
    var questionEl = document.getElementById("unclassifiedQuestion");
    if (record.question && record.question.trim()) {
      questionEl.textContent = record.question;
      wrap.classList.remove("d-none");
    } else if (wrap) {
      wrap.classList.add("d-none");
    }
  }

  /**
   * Normalizes a confidence value to a whole-number percentage. The
   * backend reports confidence in [0, 1], but this defends against a
   * value that's already 0–100 (e.g. a differently-configured provider)
   * rather than silently producing something like "9100%".
   */
  function toConfidencePercent(confidence) {
    var value = confidence <= 1 ? confidence * 100 : confidence;
    value = Math.max(0, Math.min(100, value));
    return Math.round(value);
  }
})();
