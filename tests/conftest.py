"""Pytest configuration shared by the tests in this directory."""

import os

# The navigator image has no torch or transformers. Match its mode so the tests
# import the game engine without the GPU dependencies.
os.environ.setdefault("APP_MODE", "navigator")
