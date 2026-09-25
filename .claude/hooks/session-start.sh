#!/bin/bash
# Cloud sessions start from a bare container. Install what CI installs. The
# browser for the page tests is already in the container (Playwright is pointed
# at /opt/pw-browsers), so only the Python package is needed - never run
# `playwright install` here.
set -euo pipefail
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi
python3 -c "import numpy, scipy, pandas, playwright" 2>/dev/null \
  || pip install --quiet numpy scipy pandas playwright
