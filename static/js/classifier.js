/**
 * CropCare AI — Ask page logic.
 *
 * Responsibilities:
 *  - Validate the question before submitting.
 *  - Show a live character counter.
 *  - POST { question, crop } to the Flask backend at /api/classify.
 *  - Show a loading state while waiting for a response.
 *  - Store the successful result in localStorage (both as "last result"
 *    for result.html, and appended to the full search history for
 *    history.html) and redirect to result.html.
 *  - Surface network, server, and malformed-response errors clearly.
 *
 * Security note: this file never contains, requests, or stores any LLM
 * provider API key. The browser only ever talks to this app's own Flask
 * backend (POST /api/classify) — it never calls an LLM provider directly.
 */

(function () {
  "use strict";

  var CLASSIFY_ENDPOINT = "/api/classify";
  var RESULT_STORAGE_KEY = "cropcare:lastClassification";
  var HISTORY_STORAGE_KEY = "cropcare:history";
  var HISTORY_MAX_ENTRIES = 200;
  var MAX_QUESTION_LENGTH = 500;
  var MIN_QUESTION_LENGTH = 5;

  document.addEventListener("DOMContentLoaded", function () {
    var form = document.getElementById("askForm");
    if (!form) return; // Not on the ask page.

    var cropSelect = document.getElementById("cropSelect");
    var questionText = document.getElementById("questionText");
    var charCounter = document.getElementById("charCounter");
    var questionError = document.getElementById("questionError");
    var submitBtn = document.getElementById("submitBtn");
    var submitBtnLabel = document.getElementById("submitBtnLabel");
    var submitBtnLoading = document.getElementById("submitBtnLoading");
    var submitErrorBox = document.getElementById("submitError");
    var submitErrorText = document.getElementById("submitErrorText");
    var formStatus = document.getElementById("formStatus");

    updateCharCounter();

    questionText.addEventListener("input", function () {
      updateCharCounter();
      if (questionText.value.trim().length > 0) {
        clearFieldError();
      }
    });

    form.addEventListener("submit", handleSubmit);

    /** Keep the "N / 500" counter in sync and flag when nearing the limit. */
    function updateCharCounter() {
      var length = questionText.value.length;
      charCounter.textContent = length + " / " + MAX_QUESTION_LENGTH;
      charCounter.classList.remove("cc-char-counter-warn", "cc-char-counter-limit");
      if (length >= MAX_QUESTION_LENGTH) {
        charCounter.classList.add("cc-char-counter-limit");
      } else if (length >= MAX_QUESTION_LENGTH - 50) {
        charCounter.classList.add("cc-char-counter-warn");
      }
    }

    function setFieldError(message) {
      questionError.textContent = message;
      questionText.classList.add("is-invalid");
      questionText.setAttribute("aria-invalid", "true");
    }

    function clearFieldError() {
      questionError.textContent = "";
      questionText.classList.remove("is-invalid");
      questionText.removeAttribute("aria-invalid");
    }

    function showSubmitError(message) {
      submitErrorText.textContent = message;
      submitErrorBox.classList.remove("d-none");
      announce(message);
    }

    function hideSubmitError() {
      submitErrorBox.classList.add("d-none");
      submitErrorText.textContent = "";
    }

    function announce(message) {
      formStatus.textContent = message;
    }

    function setLoading(isLoading) {
      submitBtn.disabled = isLoading;
      submitBtnLabel.classList.toggle("d-none", isLoading);
      submitBtnLoading.classList.toggle("d-none", !isLoading);
      cropSelect.disabled = isLoading;
      questionText.disabled = isLoading;
    }

    /** Basic client-side validation. The backend re-validates independently. */
    function validateQuestion(value) {
      var trimmed = value.trim();
      if (!trimmed) {
        return "Please enter a question before submitting.";
      }
      if (trimmed.length < MIN_QUESTION_LENGTH) {
        return "Your question is too short. Please add a bit more detail.";
      }
      if (trimmed.length > MAX_QUESTION_LENGTH) {
        return "Your question is too long. Please shorten it to " + MAX_QUESTION_LENGTH + " characters or fewer.";
      }
      return null;
    }

    function handleSubmit(event) {
      event.preventDefault();
      hideSubmitError();
      clearFieldError();

      var question = questionText.value;
      var crop = cropSelect.value;

      var validationMessage = validateQuestion(question);
      if (validationMessage) {
        setFieldError(validationMessage);
        questionText.focus();
        announce(validationMessage);
        return;
      }

      var payload = { question: question.trim() };
      if (crop) {
        payload.crop = crop;
      }

      setLoading(true);
      announce("Analyzing your question...");

      submitQuestion(payload)
        .then(function (result) {
          storeResultAndRedirect(result, payload);
        })
        .catch(function (err) {
          setLoading(false);
          showSubmitError(err.message);
        });
    }

    /**
     * POSTs to the Flask backend and resolves with the parsed JSON body
     * on success, or rejects with an Error carrying a farmer-friendly
     * message on any failure (network, server, or malformed response).
     */
    function submitQuestion(payload) {
      return fetch(CLASSIFY_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
        .catch(function () {
          // fetch() itself rejects on network failure (offline, DNS, CORS, etc.)
          throw new Error(
            "We couldn't reach the server. Please check your connection and try again."
          );
        })
        .then(function (response) {
          return response
            .json()
            .catch(function () {
              // Response wasn't valid JSON at all. In practice this almost
              // always means the request never reached the Flask API — e.g.
              // the frontend was opened as a local file (file://) or from a
              // static file server instead of through `python run.py`, so
              // "/api/classify" resolved to a 404/HTML page rather than this
              // app's backend. Give a message that points at that instead of
              // a generic "something went wrong".
              throw new Error(
                "Couldn't reach the classifier service at " + CLASSIFY_ENDPOINT + " " +
                "(got status " + response.status + " with a non-JSON response). " +
                "This usually means the backend isn't running — start it with " +
                "\"python run.py\" and open the app at http://localhost:5000, " +
                "rather than opening this HTML file directly."
              );
            })
            .then(function (body) {
              if (!response.ok) {
                throw new Error(errorMessageFor(response.status, body));
              }
              if (!isValidClassification(body)) {
                throw new Error(
                  "The server's response was incomplete. Please try again in a moment."
                );
              }
              return body;
            });
        });
    }

    /** Maps a non-2xx HTTP response to a clear, farmer-friendly message. */
    function errorMessageFor(status, body) {
      var serverMessage = body && typeof body.error === "string" ? body.error : null;

      if (status === 400) {
        return serverMessage || "Please check your question and try again.";
      }
      if (status === 503) {
        return "The classifier is temporarily unavailable. Please try again shortly.";
      }
      if (status >= 500) {
        return serverMessage || "Something went wrong on our end. Please try again.";
      }
      return serverMessage || "Something went wrong. Please try again.";
    }

    /** Confirms the success response has the shape the result page expects. */
    function isValidClassification(body) {
      return (
        body &&
        typeof body === "object" &&
        typeof body.category === "string" &&
        body.category.trim() !== "" &&
        typeof body.confidence === "number" &&
        typeof body.reason === "string"
      );
    }

    function storeResultAndRedirect(result, payload) {
      var record = {
        category: result.category,
        confidence: result.confidence,
        reason: result.reason,
        question: payload.question,
        crop: payload.crop || null,
        timestamp: new Date().toISOString()
      };

      try {
        localStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(record));
        appendToHistory(record);
      } catch (storageErr) {
        // localStorage can fail (private browsing, quota, disabled storage).
        // The classification itself succeeded, so tell the user rather
        // than silently failing the redirect.
        setLoading(false);
        showSubmitError(
          "We got your result but couldn't save it in this browser. " +
          "Please allow site storage and try again."
        );
        return;
      }

      window.location.href = "result.html";
    }

    /** Appends a record to the persisted search history (newest first). */
    function appendToHistory(record) {
      var history = readHistory();
      history.unshift(record);
      if (history.length > HISTORY_MAX_ENTRIES) {
        history = history.slice(0, HISTORY_MAX_ENTRIES);
      }
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
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
        return Array.isArray(parsed) ? parsed : [];
      } catch (err) {
        return [];
      }
    }
  });
})();
