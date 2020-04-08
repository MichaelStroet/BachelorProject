# Michael Stroet  11293284

import os, sys

directory = os.path.dirname(os.path.realpath(__file__))

# Add paths to the code and data folders
sys.path.append(os.path.join(directory, "code"))
sys.path.append(os.path.join(directory, "data"))

from hello import hello

if __name__ == "__main__":
    print(hello())
