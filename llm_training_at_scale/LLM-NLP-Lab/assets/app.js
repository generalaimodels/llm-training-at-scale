
(function(){
  const key = "llmnlp-theme";
  const btn = document.getElementById("themeToggle");
  if(!btn) return;

  function setTheme(mode){
    document.documentElement.dataset.theme = mode;
    try{ localStorage.setItem(key, mode); }catch(e){}
    btn.textContent = mode === "light" ? "🌙" : "☀️";
  }

  function initTheme(){
    let saved = null;
    try{ saved = localStorage.getItem(key); }catch(e){}
    if(saved === "light" || saved === "dark"){ setTheme(saved); return; }
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(prefersDark ? "dark" : "light");
  }

  btn.addEventListener("click", function(){
    const cur = document.documentElement.dataset.theme;
    setTheme(cur === "dark" ? "light" : "dark");
  });

  initTheme();
})();
