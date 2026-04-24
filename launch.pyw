"""Fizgig launcher — double-click to run without a console window.

The .pyw extension tells Windows to use pythonw.exe, which hides the
console.  Console output is captured by the GUI's built-in log — click
the status indicator in the top-right corner to view it.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHONW = os.path.join(HERE, "venv", "Scripts", "pythonw.exe")

# If the venv exists and we're not already running from it, re-launch
if os.path.exists(VENV_PYTHONW) and os.path.normcase(sys.executable) != os.path.normcase(VENV_PYTHONW):
    subprocess.Popen([VENV_PYTHONW, os.path.abspath(__file__)])
    sys.exit(0)

# Running inside venv (or no venv) — launch the GUI
sys.path.insert(0, HERE)
import runpy
runpy.run_path(os.path.join(HERE, "lora_trainer_gui.py"), run_name="__main__")
