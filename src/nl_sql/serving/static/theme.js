// Runs before first paint so a remembered light theme never flashes dark first.
// A ?theme=light or ?theme=dark link wins over the remembered choice, for sharing and screenshots.
(() => {
  const requested = new URLSearchParams(location.search).get("theme");
  let theme = "dark";
  if (requested === "light" || requested === "dark") theme = requested;
  else {
    try { theme = localStorage.getItem("nlsql-theme") === "light" ? "light" : "dark"; } catch {}
  }
  document.documentElement.dataset.theme = theme;
})();
