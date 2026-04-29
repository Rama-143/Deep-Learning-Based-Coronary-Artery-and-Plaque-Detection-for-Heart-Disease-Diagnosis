document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("uploadForm");
  const status = document.getElementById("status");

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    status.textContent = "Uploading...";

    const fd = new FormData(form);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: fd
      });

      const data = await res.json();

      if (!data.ok) {
        status.textContent = data.error || "Failed.";
        return;
      }

      status.textContent = "Done! Opening result...";
      window.location.href = data.result_url;

    } catch (err) {
      console.error(err);
      status.textContent = "Error occurred.";
    }
  });
});