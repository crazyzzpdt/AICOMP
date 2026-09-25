"""把 Markdown 转成可直接打印的 PDF，自动完成分页决策与排版。

Markdown → Pandoc → Typst → PDF 三段流水线：pypandoc-binary 捆绑的 pandoc
把 Markdown 转成 Typst 源，拼接同目录的 md2pdf_template.typ 排版规则模板后，
由 typst 包编译出 PDF。全部排版与分页规则集中在模板中：短代码块整体保持或
整块搬页、长代码块只在行边界断页、公式与图片绝不裁切、标题不孤立在页尾、
表格跨页重复表头；正文、标题与代码字体自动探测本机字体，可用 --main-font
覆盖。编译结束后打印页数、字体与排版警告，只报告不自动修改。

在项目根目录执行转换（输出省略时与输入同目录、同名 .pdf）：
    uv run python -m tools.md2pdf 文档.md
指定输出位置：
    uv run python -m tools.md2pdf 文档.md 输出/文档.pdf
直接执行脚本同样可用：
    uv run python tools/md2pdf/md2pdf.py 文档.md
换到其他项目时，把整个 md2pdf 文件夹一起复制，并在该环境安装 typst 与
pypandoc-binary（uv add typst pypandoc-binary）。
"""

from __future__ import annotations

# 内置库
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 三方库
try:
    import pypandoc
    import typst
    from typst import TypstError
except ModuleNotFoundError as error:
    # 本工具要复制到其他项目使用，缺依赖时给出安装命令而不是裸回溯。
    raise SystemExit("缺少转换依赖，请先执行：uv add typst pypandoc-binary") from error

# 自己的模块
# 无：本工具只依赖上面两个包与同目录模板，便于整体复制到其他项目。


# 排版规则模板与本脚本同目录存放，复制到其他项目时必须与脚本一起带走。
TEMPLATE_PATH: Path = Path(__file__).resolve().parent / "md2pdf_template.typ"
# 正文字体候选，按优先级探测；黑体优先、宋体兜底，均符合中文技术文档习惯。
BODY_FONT_CANDIDATES: tuple[str, ...] = ("Noto Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "SimSun")
# 标题字体候选；全部缺失时回退正文字体，标题至少与正文可辨。
HEADING_FONT_CANDIDATES: tuple[str, ...] = ("Noto Serif SC", "Noto Serif CJK SC", "SimSun")
# 代码字体候选；代码内的中文字符由模板回退到正文字体显示。
MONO_FONT_CANDIDATES: tuple[str, ...] = ("Consolas", "Cascadia Code", "Cascadia Mono", "Courier New")
# 全部标题不少于该数量才生成目录，避免一两条目录孤零零占版面。
MIN_TOC_HEADINGS: int = 2


@dataclass(frozen=True)
class ConvertConfig:
    """统一保存命令行参数；输出路径为 None 时由 convert 推导同名 PDF。"""

    source: Path
    output: Path | None
    main_font: str | None
    pandoc: Path | None


def parse_arguments(argv: list[str]) -> ConvertConfig:
    """解析输入、可选输出与两个逃生口参数。"""
    parser = argparse.ArgumentParser(description="Markdown 转 PDF，自动排版分页，输出省略时与输入同名")
    parser.add_argument("source", type=Path, help="输入的 Markdown 文件路径")
    parser.add_argument("output", nargs="?", type=Path, help="输出的 PDF 路径，省略时与输入同目录、同名 .pdf")
    parser.add_argument("--main-font", help="覆盖自动探测的正文字体名，例如 'Microsoft YaHei'")
    parser.add_argument("--pandoc", type=Path, help="指定 pandoc 可执行文件；缺省使用 pypandoc-binary 捆绑的二进制")
    return ConvertConfig(**vars(parser.parse_args(argv)))


def _heading_level(line: str) -> int:
    """返回该行 pandoc 标题的层级，非标题行返回 0。

    pandoc 输出两种标题形态：等号标题 ``= 标题`` 与带属性的
    ``#heading(level: N)``，两种都识别。
    """
    match = re.match(r"^=+\s", line)
    if match:
        return len(match.group(0)) - 1
    match = re.match(r"^#heading\(level:\s*(\d+)", line)
    return int(match.group(1)) if match else 0


def installed_font_families() -> list[str]:
    """枚举系统字体家族；枚举失败返回空列表，由调用方按缺省字体继续。"""
    try:
        return list(typst.Fonts(include_system_fonts=True).families())
    except Exception:
        return []


def detect_font(candidates: tuple[str, ...], families: list[str]) -> str | None:
    """返回候选列表中第一个已安装的字体家族，全部缺失时返回 None。"""
    for name in candidates:
        if name in families:
            return name
    return None


def build_typst_source(typst_body: str) -> tuple[str, bool]:
    """拼接规则模板并决定目录，返回最终 Typst 源与是否生成目录。

    目录在全部标题达到 MIN_TOC_HEADINGS 时插入到首个一级标题之后，
    没有一级标题时放在正文最前。
    """
    lines = typst_body.splitlines()
    has_outline = sum(1 for line in lines if _heading_level(line) >= 1) >= MIN_TOC_HEADINGS
    if has_outline:
        insert_at = next((i + 1 for i, line in enumerate(lines) if _heading_level(line) == 1), 0)
        lines.insert(insert_at, "#outline(title: [目录], depth: 3)")
    # show rule 必须位于正文之前才会作用于后文，模板固定拼在最前。
    return TEMPLATE_PATH.read_text(encoding="utf-8") + "\n" + "\n".join(lines) + "\n", has_outline


def count_pages(pdf_data: bytes) -> int | None:
    """按 PDF 页对象计数统计页数，失败返回 None 不影响主流程。

    typst-py 0.15 的 query 不再支持 <page> 选择器（返回空列表），改从生成
    的 PDF 字节里数 /Type /Page 对象；\\b 词边界排除 /Type /Pages 页面树
    节点与 /Type /PageLabel 页码标签对象。
    """
    try:
        matches = re.findall(rb"/Type\s*/Page\b", pdf_data)
        return len(matches) if matches else None
    except Exception:
        return None


def convert(config: ConvertConfig) -> None:
    """执行转换并打印自动体检报告：字体、页数、目录与排版警告。

    Args:
        config: 已解析的命令行参数。

    Raises:
        FileNotFoundError: 输入、模板或 --pandoc 指定文件缺失。
        ValueError: 输入不是 Markdown，或输出路径与输入相同。
        RuntimeError: pandoc 不可用或转换失败。
        TypstError: Typst 编译失败，错误信息与提示由入口打印。
    """
    source = config.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入的 Markdown 文件：{source}")
    if source.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"输入必须是 .md 或 .markdown 文件：{source}")
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"缺少排版规则模板：{TEMPLATE_PATH}；请与 md2pdf.py 一起复制")
    output = (config.output or source.with_suffix(".pdf")).expanduser().resolve()
    if output == source:
        raise ValueError("输出 PDF 不能与输入 Markdown 同名，防止覆盖原文档")
    if config.pandoc is not None:
        if not config.pandoc.is_file():
            raise FileNotFoundError(f"找不到 pandoc 可执行文件：{config.pandoc}")
        # PYPANDOC_PANDOC 是 pypandoc 官方支持的覆盖入口，必须在转换调用前设置。
        os.environ["PYPANDOC_PANDOC"] = str(config.pandoc)
    try:
        pandoc_version = pypandoc.get_pandoc_version()
        typst_body = pypandoc.convert_file(str(source), "typst", format="markdown")
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"pandoc 不可用或转换失败：{error}；请安装 pypandoc-binary，或用 --pandoc 指定路径") from error
    print(f"pandoc {pandoc_version} 已把 Markdown 转成 Typst 源（{pypandoc.get_pandoc_path()}）")

    families = installed_font_families()
    body_font = config.main_font or detect_font(BODY_FONT_CANDIDATES, families)
    heading_font = detect_font(HEADING_FONT_CANDIDATES, families) or body_font
    mono_font = detect_font(MONO_FONT_CANDIDATES, families)
    # 模板自带缺省字体，只把探测到的字体经 sys.inputs 传入覆盖。
    sys_inputs: dict[str, str] = {}
    for key, value in (("body-font", body_font), ("heading-font", heading_font), ("mono-font", mono_font)):
        if value:
            sys_inputs[key] = value

    typst_source, has_outline = build_typst_source(typst_body)
    # 时间戳取输入文件修改时间，同一 Markdown 重复转换得到相同 PDF，便于迭代对比。
    # 源码必须编码成 bytes：typst-py 把 str 输入当作 .typ 文件路径而非源码文本。
    data, warnings = typst.compile_with_warnings(typst_source.encode("utf-8"), output=None, root=str(source.parent),
                                                 sys_inputs=sys_inputs, timestamp=int(source.stat().st_mtime))
    if not isinstance(data, bytes):
        raise RuntimeError("Typst 未返回 PDF 内容")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)

    pages = count_pages(data)
    print(f"转换完成：{output}")
    print(f"共 {pages} 页" if pages is not None else "页数统计失败，不影响 PDF 内容")
    if not body_font:
        print("警告：未找到常见中文字体，中文可能显示异常；请安装中文字体或用 --main-font 指定")
    else:
        print(f"正文字体：{body_font}；标题字体：{heading_font or '同正文'}；代码字体：{mono_font or '默认等宽'}")
    if has_outline:
        print("已自动生成目录")
    if warnings:
        print(f"排版警告 {len(warnings)} 条：")
        for warning in warnings:
            print(f"  - {warning.message}")
    else:
        print("排版警告 0 条")


def run(argv: list[str]) -> None:
    """解析参数并执行转换，把预期错误转成可读提示与非零退出码。"""
    try:
        convert(parse_arguments(argv))
    except TypstError as error:
        print(f"Typst 编译失败：{error.message}", file=sys.stderr)
        for hint in error.hints:
            print(f"提示：{hint}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"转换失败：{error}", file=sys.stderr)
        sys.exit(1)


# Windows 下被其他模块导入时不应自动转换，直接执行脚本时从本入口进入。
if __name__ == "__main__":
    run(sys.argv[1:])
