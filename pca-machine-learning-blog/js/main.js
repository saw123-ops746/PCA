// Table of contents, scroll-spy and in-page navigation (no framework).
(function () {
  var secs = Array.prototype.slice.call(document.querySelectorAll("article > section[id]"));
  var toc = document.getElementById("toc-links");
  secs.forEach(function (s) {
    var h = s.querySelector("h2"); if (!h) return;
    var a = document.createElement("a"); a.href = "#" + s.id; a.textContent = h.textContent; toc.appendChild(a);
  });
  var links = toc.querySelectorAll("a");
  function spy() {
    var y = window.scrollY + 120, cur = secs[0];
    secs.forEach(function (s) { if (s.offsetTop <= y) cur = s; });
    links.forEach(function (a) { a.classList.toggle("on", a.getAttribute("href") === "#" + cur.id); });
  }
  window.addEventListener("scroll", spy, { passive: true }); spy();
  // Move keyboard focus to the target section after in-page navigation.
  document.addEventListener("click", function (e) {
    var a = e.target.closest('a[href^="#"]'); if (!a) return;
    var t = document.getElementById(a.getAttribute("href").slice(1)); if (!t) return;
    t.setAttribute("tabindex", "-1"); setTimeout(function () { t.focus({ preventScroll: true }); }, 400);
  });
  // Expand / collapse all <details> in the interview section.
  var btn = document.getElementById("toggle-qa");
  if (btn) btn.addEventListener("click", function () {
    var items = document.querySelectorAll("#interview details"); var open = btn.dataset.open !== "1";
    items.forEach(function (d) { d.open = open; }); btn.dataset.open = open ? "1" : "0";
    btn.textContent = open ? "Collapse all answers" : "Expand all answers";
  });
})();
