# conftest.py
# Makes 'src' importable when running pytest from the project root.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))