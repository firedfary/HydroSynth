import os
import sys

# 自动设置路径
_proj_root = os.path.dirname(__file__)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
