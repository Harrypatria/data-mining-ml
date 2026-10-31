export type Theme = "light" | "dark";

/** First visit follows prefers-color-scheme; a choice is kept in localStorage when storage is available. */
export function currentTheme(): Theme {
  try {
    const saved = localStorage.getItem("ml-theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage blocked: fall through to the system preference
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Apply a theme; persist only an explicit choice so first visits keep following the system setting. */
export function applyTheme(theme: Theme, persist = false): Theme {
  document.documentElement.classList.toggle("dark", theme === "dark");
  if (persist) {
    try {
      localStorage.setItem("ml-theme", theme);
    } catch {
      // storage blocked: the theme still applies for this visit
    }
  }
  return theme;
}
