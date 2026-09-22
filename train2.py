"""D-FINE 独立训练入口。

所有 D-FINE 源码位于 ``src/D-FINE``；本入口只负责把命令行参数转交给
官方 ``train.py``，不导入 YOLO 训练器或 ``src/yolo``。

示例：
    uv run python train2.py -c src/D-FINE/configs/dfine/include/dfine_hgnetv2.yml --output-dir runs/dfine
    uv run python train2.py -c <配置文件> -r <D-FINE检查点>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent
DFINE_ROOT: Path = PROJECT_ROOT / "src" / "D-FINE"


if __name__ == "__main__":
    if not (DFINE_ROOT / "train.py").is_file():
        raise FileNotFoundError(f"缺少D-FINE源码：{DFINE_ROOT}")
    sys.path.insert(0, str(DFINE_ROOT))
    runpy.run_path(str(DFINE_ROOT / "train.py"), run_name="__main__")
