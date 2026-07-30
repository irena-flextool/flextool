"""``python -m flextool.calibrate`` entry point.

Delegates to :func:`flextool.calibrate._cli.main` so the calibrator can be
launched as a module.
"""

from __future__ import annotations

from flextool.calibrate._cli import main

raise SystemExit(main())
