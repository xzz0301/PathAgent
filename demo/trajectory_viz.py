"""Compatibility import for historical visualization scripts; implementation: trajectory.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory import TraceRecorder, export_trajectory_artifacts, split_zoom_patch, main

if __name__ == "__main__":
    main()
