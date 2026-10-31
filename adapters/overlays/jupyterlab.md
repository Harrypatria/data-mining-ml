## Environment: JupyterLab 4
- The ipywidgets 8 widget manager is built in; no extension install is needed for core widgets.
- Headings feed the table of contents, so write them as findings.
- Pair notebooks with jupytext (`jupytext --set-formats ipynb,py:percent`) so diffs and reviews happen on the text file.
- Clean-run check: Kernel, Restart Kernel and Run All Cells, or `jupyter nbconvert --to notebook --execute`.
