## Environment: Google Colab with Gemini
- First code cell: detect Colab with `"google.colab" in sys.modules`, then `%pip install -q` pinned packages and the project package from its tagged repository.
- Read keys with `from google.colab import userdata; userdata.get("NAME")`; if a secret is missing, explain how to add it under Secrets (key icon) and continue with the offline path.
- Mount Drive only when the user asks: `from google.colab import drive; drive.mount("/content/drive")`.
- The runtime is ephemeral: save models, artifacts and reports to Drive or offer `files.download`.
- Core ipywidgets work as is; call `output.enable_custom_widget_manager()` only for third-party widgets.
- Use `%load_ext google.colab.data_table` for large frame previews instead of printing frames.
- Suggest a GPU or high-RAM runtime only when the workload needs it (deep learning, very large data).
- When generating cells, keep each cell to one step, explain it in one sentence of Markdown above it, and never invent output values in the Markdown.
