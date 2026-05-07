"""API package init.

Best-effort defense-in-depth for macOS fork-safety env vars. The authoritative
place to set these is the shell, *before* Python starts (see
`scripts/run_server.sh`). Apple's Objective-C runtime reads
OBJC_DISABLE_INITIALIZE_FORK_SAFETY at process startup; setting it from Python
is normally too late, but `setdefault` here covers cases where Python re-execs
or where a teammate launches uvicorn from an editor without the wrapper script.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
