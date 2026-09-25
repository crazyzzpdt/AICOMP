"""包入口：uv run python -m tools.md2pdf 与直接执行 md2pdf.py 等价。"""

import sys

# 自己的模块
from tools.md2pdf.md2pdf import run

run(sys.argv[1:])
