"""整合复赛提交包：按附件2目录生成"参赛团队编号-团队名称-复赛-队长姓名"文件夹。

按官方《附件2》要求整合三部分：技术方案PDF、模型文件、代码与数据。代码与
数据目录放入排行榜提交ZIP、docs/src/tools三个目录、四个根入口脚本、
pyproject.toml和uv.lock，以及模型评估目录（--run 指定运行目录的内容，剔除
code/weights：源码快照与权重已分别随代码目录和模型文件提交）；--report 指定
的技术方案 Markdown 经 tools/md2pdf 转成 PDF 放入根目录。目录与文件只复制
代码和文档类（跳过视频、PDF、缓存），符合附件2大文件不放入代码目录的要求。

目标文件夹由本工具固定命名并整包重建：重复执行先删除旧的再重新生成，只删
本工具自己的目标文件夹，不动打包位置下的其他内容。全部输入先校验、技术方
案先转成临时PDF，任一失败都不破坏旧的已整合提交包。--model、--submission、
--run、--report 为可选参数，未提供的部分跳过并打印提示。代码与数据目录还会经
uv export 从 uv.lock 生成 requirements.txt，满足赛题规则复赛清单要求。

在项目根目录执行（打包位置默认项目根目录）：
    uv run python tools/IntegrateAndPackage.py --model 权重路径 --submission submission.zip路径 --run runs/detect/版本运行目录 --report docs/技术方案.md
指定打包位置（文件夹将创建在该目录下）：
    uv run python tools/IntegrateAndPackage.py 打包目录 --model 权重路径 --submission submission.zip路径 --run runs/detect/版本运行目录 --report docs/技术方案.md
"""

from __future__ import annotations

# 内置库
import argparse
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

# 直接以脚本方式执行时 sys.path 只含脚本所在目录，先补上项目根目录，
# 后面的 tools.md2pdf 导入才找得到包；用 -m 方式执行时 __package__ 非空，跳过。
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 三方库
# 无：打包本身只用标准库，PDF 转换复用 tools.md2pdf 的能力。

# 自己的模块
from tools.md2pdf.md2pdf import ConvertConfig, TypstError, convert as convert_md_to_pdf


# 相对路径以入口文件所在目录为基准，兼容从其他位置启动。
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# 报名系统截图确认的参赛团队信息，来源 docs/预测与赛事提交.md。
TEAM_ID: str = "AIC-2026-81588292"
TEAM_NAME: str = "牛副队"
CAPTAIN: str = "潘炜德"
# 附件2要求的根文件夹命名：参赛团队编号-团队名称-复赛-队长姓名。
PACKAGE_NAME: str = f"{TEAM_ID}-{TEAM_NAME}-复赛-{CAPTAIN}"
# 只复制代码与文档类文件；.typ 是 md2pdf 的排版模板，必须随工具一起打包；
# docs 下的视频、PDF 等大文件与缓存不进入代码目录。
COPY_SUFFIXES: frozenset[str] = frozenset({".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".typ"})
COPY_NAMES: frozenset[str] = frozenset({"LICENSE", "NOTICE"})
# 跳过版本控制、字节码缓存与虚拟环境，避免把运行垃圾复制进提交包。
SKIP_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".venv"})


@dataclass(frozen=True)
class PackageConfig:
    """统一保存命令行参数；未提供的文件对应部分跳过并列入待完善清单。"""

    package_dir: Path | None
    model: Path | None
    submission: Path | None
    run: Path | None
    report: Path | None


def parse_arguments(argv: list[str]) -> PackageConfig:
    """解析打包位置与三个可选文件路径，未传的参数保留为 None。"""
    parser = argparse.ArgumentParser(description="按附件2目录整合复赛提交包")
    parser.add_argument("package_dir", nargs="?", type=Path, help="打包位置，默认项目根目录；文件夹将创建在该目录下")
    parser.add_argument("--model", type=Path, help="模型权重文件路径，复制到 编号-模型文件/ 目录")
    parser.add_argument("--submission", type=Path, help="排行榜提交的 submission.zip 路径，复制到 代码与数据/ 目录")
    parser.add_argument("--run", type=Path, help="运行目录路径，其内容（剔除code/weights）复制到 代码与数据/模型评估/ 目录")
    parser.add_argument("--report", type=Path, help="技术方案 Markdown 路径，转成 PDF 后放到根目录")
    return PackageConfig(**vars(parser.parse_args(argv)))


def copy_source_files(source: Path, target: Path) -> int:
    """按扩展名过滤复制代码与文档，跳过缓存目录，返回复制的文件数。"""
    count = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink() or any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_file() and (path.suffix.lower() in COPY_SUFFIXES or path.name in COPY_NAMES):
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)  # copy2 保留修改时间等元数据
            count += 1
    return count


def package(config: PackageConfig) -> None:
    """在打包位置重建附件2目录，复制代码与数据，并按需生成技术方案PDF。

    Args:
        config: 已解析的命令行参数。

    Raises:
        FileNotFoundError: 打包位置、模型权重、运行目录或待打包源文件缺失。
        ValueError: 提交ZIP无效或模型不是 .pt/.pth 权重文件。
        RuntimeError: uv 命令缺失或生成 requirements.txt 失败。
        TypstError: 技术方案转换失败，错误信息与提示由入口打印。
    """
    package_dir = (config.package_dir or PROJECT_ROOT).expanduser().resolve()
    if not package_dir.is_dir():
        raise FileNotFoundError(f"打包位置不存在：{package_dir}")
    root = package_dir / PACKAGE_NAME

    # 先校验全部输入，任一无效时直接报错，不删除旧的已整合提交包。
    model = config.model.expanduser().resolve() if config.model is not None else None
    if model is not None:
        if not model.is_file():
            raise FileNotFoundError(f"找不到模型权重文件：{model}")
        if model.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError(f"模型权重必须是 .pt 或 .pth 文件：{model}")
    submission = config.submission.expanduser().resolve() if config.submission is not None else None
    if submission is not None and (not submission.is_file() or not zipfile.is_zipfile(submission)):
        raise ValueError(f"提交结果不是有效的 ZIP 文件：{submission}")
    run = config.run.expanduser().resolve() if config.run is not None else None
    if run is not None and not run.is_dir():
        raise FileNotFoundError(f"找不到运行目录：{run}")
    report = config.report.expanduser().resolve() if config.report is not None else None

    # 技术方案先转成临时PDF：转换失败时不破坏旧的已整合提交包；
    # 临时文件无论后续复制成功还是中途失败都由 finally 清理。
    pdf_temporary: Path | None = None
    try:
        if report is not None:
            pdf_temporary = package_dir / f".{PACKAGE_NAME}-技术方案.pdf.partial"
            convert_md_to_pdf(ConvertConfig(source=report, output=pdf_temporary, main_font=None, pandoc=None))

        # 目标文件夹由本工具固定命名并整包重建，重复执行先删除旧结果再重新生成。
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            pass
        code_dir = root / f"{TEAM_ID}-代码与数据"
        model_dir = root / f"{TEAM_ID}-模型文件"
        code_dir.mkdir(parents=True)
        model_dir.mkdir()

        pending: list[str] = []
        # 模型、提交ZIP与技术方案为可选参数，缺失时跳过并打印提示，不冒充完整交付。
        if model is None:
            pending.append("缺少 --model 模型权重，模型文件目录为空")
        else:
            shutil.copy2(model, model_dir / model.name)
        if submission is None:
            pending.append("缺少 --submission 提交ZIP，代码与数据目录不含排行榜结果")
        else:
            shutil.copy2(submission, code_dir / submission.name)
        # 模型评估：--run 指定运行目录的内容整体复制进 代码与数据/模型评估/，
        # 剔除 code/weights——源码快照与权重已分别随代码目录和模型文件提交。
        if run is None:
            pending.append("缺少 --run 运行目录，代码与数据目录没有模型评估内容")
        else:
            shutil.copytree(run, code_dir / "模型评估",
                            copy_function=shutil.copy2,
                            ignore=shutil.ignore_patterns("code", "weights", "__pycache__", ".git"))
            print(f"已复制模型评估内容：{code_dir / '模型评估'}")

        # 代码与数据：四个根入口、依赖声明与 docs/src/tools 三个目录。
        for name in ("predict1.py", "predict2.py", "train1.py", "train2.py", "pyproject.toml", "uv.lock"):
            path = PROJECT_ROOT / name
            if not path.is_file():
                raise FileNotFoundError(f"项目根目录缺少待打包文件：{path}")
            shutil.copy2(path, code_dir / name)
        # 赛题规则复赛清单要求 requirements.txt：由 uv export 从 uv.lock 导出
        # （不含 dev 组），与 pyproject.toml 声明版本一致，不额外联网解析。
        # 显式指定 UTF-8 解码：默认 GBK 会因 uv 输出的 UTF-8 字符报解码错误。
        try:
            export = subprocess.run(
                ["uv", "export", "--format", "requirements-txt",
                 "--output-file", str(code_dir / "requirements.txt")],
                cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise RuntimeError("找不到 uv 命令，无法生成 requirements.txt") from None
        if export.returncode != 0:
            raise RuntimeError(f"生成 requirements.txt 失败：{export.stderr.strip()}")
        print(f"已生成 requirements.txt：{code_dir / 'requirements.txt'}")
        for name in ("docs", "src", "tools"):
            count = copy_source_files(PROJECT_ROOT / name, code_dir / name)
            print(f"已复制 {name}/ 下 {count} 个代码与文档文件")

        if pdf_temporary is not None:
            shutil.copy2(pdf_temporary, root / f"{TEAM_ID}-技术方案.pdf")
            print(f"技术方案已转成 PDF：{root / f'{TEAM_ID}-技术方案.pdf'}")
        else:
            pending.append("缺少 --report 技术方案Markdown，根目录没有技术方案PDF")

        print(f"复赛提交包已整合：{root}")
        for item in pending:
            print(f"提示：{item}")
    finally:
        if pdf_temporary is not None and pdf_temporary.exists():
            pdf_temporary.unlink()


# Windows 下被其他模块导入时不应自动打包，正式入口放在保护块内。
if __name__ == "__main__":
    try:
        package(parse_arguments(sys.argv[1:]))
    except TypstError as error:
        print(f"技术方案转换失败：{error.message}", file=sys.stderr)
        for hint in error.hints:
            print(f"提示：{hint}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"打包失败：{error}", file=sys.stderr)
        sys.exit(1)
