import sys
if sys.prefix != sys.base_prefix:
    print(f"Running in a virtual environment. Path: {sys.prefix}")
    