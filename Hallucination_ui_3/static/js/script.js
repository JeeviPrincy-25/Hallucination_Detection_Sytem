// ============================================================
//  HALLUCINATION DETECTION SYSTEM — script.js
// ============================================================

// -----------------------------
// SHOW LOADER ON FORM SUBMIT
// -----------------------------
document.addEventListener("DOMContentLoaded", () => {

    const form   = document.getElementById("analyzeForm");
    const loader = document.getElementById("loader");

    if (form && loader) {
        form.addEventListener("submit", () => {
            loader.style.display = "flex";
        });
    }

});


// -----------------------------
// BUTTON RIPPLE EFFECT
// -----------------------------
document.addEventListener("click", function(e) {

    const btn = e.target.closest(".primary-btn");
    if (!btn) return;

    // remove old ripple
    const old = btn.querySelector(".ripple");
    if (old) old.remove();

    const circle   = document.createElement("span");
    const diameter = Math.max(btn.clientWidth, btn.clientHeight);
    const radius   = diameter / 2;
    const rect     = btn.getBoundingClientRect();

    circle.style.width  = circle.style.height = `${diameter}px`;
    circle.style.left   = `${e.clientX - rect.left  - radius}px`;
    circle.style.top    = `${e.clientY - rect.top   - radius}px`;
    circle.classList.add("ripple");

    btn.appendChild(circle);
});
