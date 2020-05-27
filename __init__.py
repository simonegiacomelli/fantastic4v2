from datetime import datetime
from pathlib import Path

now = datetime.now()

import sys


def add_sys_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


add_sys_path(str(Path(__file__).parent))
print(sys.path)
