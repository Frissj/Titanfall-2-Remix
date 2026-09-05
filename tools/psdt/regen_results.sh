#!/bin/sh
# Regenerate tools/psdt/RESULTS.txt. Run from the repository root.
{
  echo "PSDT test suite output. Regenerate with:"
  echo "    tools/psdt/regen_results.sh"
  echo
  echo "Consistency check (shader / header / C++ agreement):"
  python3 tools/psdt/psdt_check.py
  echo
  python3 tools/psdt/psdt_suite.py
} > tools/psdt/RESULTS.txt 2>&1
