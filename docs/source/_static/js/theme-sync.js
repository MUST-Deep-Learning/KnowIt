document.addEventListener("DOMContentLoaded", () => {
    // Get the saved theme from localStorage (default to "auto")
    const savedTheme = localStorage.getItem("__theme") || "auto";
    applyTheme(savedTheme);

    // Find the theme toggle button and attach event listener
    const themeToggleButton = document.querySelector("[data-theme-toggle]");
    if (themeToggleButton) {
        themeToggleButton.addEventListener("click", () => {
            // Get current theme and toggle it
            const currentTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = currentTheme === "light" ? "dark" : "light";

            // Apply the new theme
            applyTheme(newTheme);

            // Save to localStorage
            localStorage.setItem("__theme", newTheme);
        });
    }
});

// Function to apply the theme correctly
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    document.body.classList.remove("light", "dark");
    document.body.classList.add(theme);  // Some themes depend on body classes
}
