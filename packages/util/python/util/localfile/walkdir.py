import os
from typing import Callable, List


def walkdir(
    dir: str,
    callback: Callable[[str, List[str], List[str]], None]
):
    """
    Walk through the directory tree rooted at dir, calling the callback function
    with the root directory, directories, and files at each level.

    :param dir: The root directory to walk through.
    :param callback: A function that takes three arguments: the root directory,
                     a list of directories, and a list of files.
    """
    for root, dirs, files in os.walk(dir):
        callback(root, dirs, files)
