"""mlkit: importable atoms shared by the notebooks, the FastAPI service and the tests.

Custom transformers live here (never in a notebook's __main__) so a pickled pipeline
can be loaded by any process that installs this package.
"""

__version__ = "1.0.0"
