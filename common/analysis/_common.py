"""Shared setup for the analysis scripts: puts common/ on sys.path so they can
be run from any working directory."""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(_here, ".."))
REPO = os.path.abspath(os.path.join(COMMON, ".."))
if COMMON not in sys.path:
    sys.path.insert(0, COMMON)
