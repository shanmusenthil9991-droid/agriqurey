/**
 * CropCare AI — Home page interactions
 * Handles: active nav-link highlighting on scroll, collapsing the
 * mobile menu after a link is tapped, and the sticky-navbar shadow.
 */

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    var navbar = document.querySelector(".cc-navbar");
    var navLinks = Array.prototype.slice.call(
      document.querySelectorAll(".cc-navbar .nav-link[href^='#']")
    );
    var sections = navLinks
      .map(function (link) {
        var id = link.getAttribute("href").slice(1);
        return document.getElementById(id);
      })
      .filter(Boolean);

    var navMenuEl = document.getElementById("ccNavMenu");
    var bsCollapse =
      navMenuEl && window.bootstrap
        ? window.bootstrap.Collapse.getOrCreateInstance(navMenuEl, { toggle: false })
        : null;

    // Collapse the mobile menu after choosing a link.
    navLinks.forEach(function (link) {
      link.addEventListener("click", function () {
        if (bsCollapse && navMenuEl.classList.contains("show")) {
          bsCollapse.hide();
        }
      });
    });

    // Add a subtle shadow to the navbar once the page has scrolled.
    function onScroll() {
      if (!navbar) return;
      if (window.scrollY > 8) {
        navbar.classList.add("cc-navbar-scrolled");
      } else {
        navbar.classList.remove("cc-navbar-scrolled");
      }
      setActiveLink();
    }

    // Highlight the nav link for the section currently in view.
    function setActiveLink() {
      var scrollPos = window.scrollY + 120;
      var currentId = sections.length ? sections[0].id : null;

      sections.forEach(function (section) {
        if (section.offsetTop <= scrollPos) {
          currentId = section.id;
        }
      });

      navLinks.forEach(function (link) {
        var isActive = link.getAttribute("href") === "#" + currentId;
        link.classList.toggle("active", isActive);
        if (isActive) {
          link.setAttribute("aria-current", "page");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  });
})();
