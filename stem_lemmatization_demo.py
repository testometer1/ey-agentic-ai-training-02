#!/usr/bin/env python3  # Tells the computer: run this file with Python 3
"""
Demonstrate the difference between Stemming and Lemmatization with NLTK.

This script:
1. Checks whether the required Python package(s) are installed.
2. Installs missing packages automatically with pip.
3. Downloads the needed NLTK data automatically.
4. Applies stemming and lemmatization to the exact 10 words requested.
5. Prints a clean table and a short explanation.
"""  # This block is a description of the whole file; Python does not run it as code

import importlib.util  # Tool to check: "Is this Python package already installed?"
import subprocess  # Tool to run another program from Python (we use it to run pip)
import sys  # Gives access to the current Python program (which Python is running this file)


def ensure_package(package_name):  # Create a helper named ensure_package that takes a package name
    """Install a Python package if it is missing."""  # Short note about what this helper does
    if importlib.util.find_spec(package_name) is None:  # If the package is NOT found on this computer...
        print(f"Installing missing package: {package_name}")  # Tell the user we are about to install it
        try:  # Try the next step; if it fails, jump to "except"
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])  # Run: python -m pip install <package>
        except subprocess.CalledProcessError as exc:  # If that install command failed...
            print(f"Failed to install {package_name}.")  # Tell the user the install did not work
            raise SystemExit(1) from orig_exc if False else SystemExit(1)  # placeholder to keep structure
    else:  # Otherwise, the package IS already installed...
        print(f"Package already available: {package_name}")  # Tell the user we can skip installing it
