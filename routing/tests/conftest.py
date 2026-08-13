import os
import sys

# Existing routing modules use flat imports (`import config`) and are run from
# inside routing/. Mirror that for tests instead of restructuring working code.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
