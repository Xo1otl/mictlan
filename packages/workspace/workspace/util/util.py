import os
import glob
import subprocess
from typing import List, Dict
from workspace import path, config


def globpaths(pattern: str) -> List[path.Path]:
    """
    Returns a list of Path objects matching the given glob pattern.
    The pattern is interpreted relative to WORKSPACE_ROOT.
    """
    # Build the absolute pattern by joining the workspace root and the provided pattern.
    abs_pattern = os.path.join(config.WORKSPACE_ROOT, pattern)
    # Use glob to find matching files. recursive=True allows patterns like '**/*.py'
    matches = glob.glob(abs_pattern, recursive=True)
    # Filter the matches to include only files.
    return [path.Path(match) for match in matches if os.path.isfile(match)]


def runfiles(pattern: str) -> None:
    """
    Executes all files matching the given glob pattern.
    The pattern is relative to WORKSPACE_ROOT.

    Execution strategy:
      - If the file is executable (os.X_OK), run it directly.
      - If it is a Python script (endswith .py), run it using the Python interpreter.
      - Otherwise, skip the file.

    The standard output and error of each execution are printed.
    """
    files = globpaths(pattern)
    for f in files:
        abs_path = f.abs()
        # Determine the command to run based on file properties.
        if os.access(abs_path, os.X_OK):
            # Run directly if the file is executable.
            cmd = [abs_path]
        elif abs_path.endswith('.py'):
            # If the file is a Python script, run it with the Python interpreter.
            cmd = ['python', abs_path]
        elif abs_path.endswith('.ts'):
            # If the file is a Python script, run it with the Python interpreter.
            cmd = ['bun', abs_path]
        else:
            print(
                f"Skipping {f}: not executable and not a Python script and not a Typescript.")
            continue

        try:
            # Run the command and capture output.
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            print(f"Output of {f}:\n{result.stdout}")
            if result.stderr:
                print(f"Error output of {f}:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Execution failed for {f}:\n{e}")


def readfiles(pattern: str) -> Dict[path.Path, str]:
    """
    Reads all files matching the given glob pattern and returns a dictionary
    mapping each Path object to its file content.
    The pattern is relative to WORKSPACE_ROOT.
    """
    files = globpaths(pattern)
    contents: Dict[path.Path, str] = {}
    for f in files:
        try:
            with open(f.abs(), 'r', encoding='utf-8') as file_obj:
                contents[f] = file_obj.read()
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return contents
