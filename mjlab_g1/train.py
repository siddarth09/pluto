"""mjlab train entrypoint that registers the PLUTO tasks first.

The registry is a module-level dict populated at import time, so importing
pluto.mjlab_g1 before calling mjlab's own train makes the tasks resolvable without
modifying mjlab.

    python -m pluto.mjlab_g1.train --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pluto.mjlab_g1  # noqa: F401  registers the tasks
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
