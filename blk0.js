
window.__showErr = msg => {
  const el = document.getElementById('err');
  if (el) { el.textContent = String(msg); el.style.display = 'block'; }
};
window.addEventListener('error', ev => window.__showErr('錯誤：' + (ev.message || ev.error)));
window.addEventListener('unhandledrejection', ev => window.__showErr('錯誤：' + ((ev.reason && ev.reason.message) || ev.reason)));
