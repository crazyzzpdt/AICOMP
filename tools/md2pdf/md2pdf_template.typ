// md2pdf 排版规则模板：与同目录 md2pdf.py 配套，由脚本拼接在 pandoc 生成的
// Typst 源文件头部。页面样式与分页规则全部集中在此文件，修改排版只改这里；
// 字体等参数经 sys.inputs 由 Python 端探测后传入。

// 一、字体与页面：缺省值保证模板单独可编译，实际值由脚本传入
#let body-font = sys.inputs.at("body-font", default: "Noto Sans SC")
#let heading-font = sys.inputs.at("heading-font", default: "Noto Serif SC")
#let mono-font = sys.inputs.at("mono-font", default: "Consolas")

// A4、四边 2.5 厘米、页脚居中页码；正文中文两端对齐并放宽行距
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(font: (body-font, heading-font), size: 10.5pt, lang: "zh")
#set par(justify: true, leading: 0.85em)

// 二、标题：sticky 是 Typst 官方防止标题孤立在页尾的机制，标题会随正文
// 一起搬到下一页顶部；衬线字体与正文形成层级对比
#show heading: it => block(sticky: true, it)
#show heading.where(level: 1): set text(size: 17pt, weight: "bold", font: heading-font)
#show heading.where(level: 2): set text(size: 14pt, weight: "bold", font: heading-font)
#show heading.where(level: 3): set text(size: 12pt, weight: "bold", font: heading-font)
#show heading.where(level: 4): set text(size: 11pt, weight: "bold", font: heading-font)

// 三、代码：0.15 起 raw 不再有 fill/inset/radius 参数，底色、内边距与圆角
// 改由 show 规则包一层 block；行内代码只加左右小内边距。代码块不超过 12 行
// 整体排版，放不下时整块搬到下一页，绝不出现行间切割；更长代码允许跨页，
// Typst 只在行边界断页、不切断单行。代码内的中文字符回退到正文字体显示。
#show raw: set text(font: (mono-font, body-font), size: 9pt)
#show raw.where(block: false): box.with(fill: luma(94%), inset: (x: 3pt, y: 1pt), radius: 2pt)
#show raw.where(block: true): it => {
  let lines = it.text.split("\n").len()
  block(fill: luma(94%), inset: (x: 8pt, y: 5pt), radius: 3pt, breakable: lines > 12, it)
}

// 四、表格：pandoc 输出 table.header 表头，0.15 起其 repeat 参数默认为
// true，跨页自动重复表头，无需额外规则；表格包在 kind 为 table 的
// figure 中，保持其可跨页，不包成不可断块

// 五、图片：fit contain 等比缩放、绝不裁切；带图注的图片整块搬页，
// 图注不与图片分离（表格图注例外，见上）
#show image: set image(fit: "contain")
#show figure: it => {
  if it.kind == "table" {
    it
  } else {
    block(breakable: false, it)
  }
}

// 公式无需规则：Typst 的显示公式是单行不可断块，天然整体搬页；
// 过宽公式会以排版警告形式出现在转换报告的体检结果中。
