/** 极简 markdown → HTML。
 *
 * 为什么不用 react-markdown:模型输出的 markdown 很有限(表格/加粗/代码/列表),
 * 自己实现只要几十行、零依赖、流式渲染时更快。若将来要支持公式/图表再换库。
 *
 * 安全:【先转义再套标签】—— 模型输出不可全信,避免 XSS。
 */

function escapeHtml(s: string): string {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]!);
}

/** markdown 表格块 → <table> */
function renderTable(rows: string[]): string {
  const cells = rows
    .filter((l) => !/^\s*\|[\s|:-]+\|\s*$/.test(l)) // 丢掉 |---|---| 分隔行
    .map((l) =>
      l
        .trim()
        .replace(/^\||\|$/g, "")
        .split("|")
        .map((c) => c.trim()),
    );
  if (!cells.length) return "";
  const head = cells.shift()!;
  return (
    "<table><thead><tr>" +
    head.map((c) => `<th>${c}</th>`).join("") +
    "</tr></thead><tbody>" +
    cells.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") +
    "</tbody></table>"
  );
}

export function renderMarkdown(src: string): string {
  const inline = escapeHtml(src)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/^#{1,4}\s+(.+)$/gm, "<strong>$1</strong>")
    .replace(/^[-*]\s+(.+)$/gm, "· $1")
    .replace(/^\s*---+\s*$/gm, "");

  // 连续的 |...| 行聚成一个表格块
  const out: string[] = [];
  let block: string[] = [];
  const flush = () => {
    if (block.length) {
      out.push(renderTable(block));
      block = [];
    }
  };
  for (const line of inline.split("\n")) {
    if (/^\s*\|.*\|\s*$/.test(line)) block.push(line);
    else {
      flush();
      out.push(line);
    }
  }
  flush();
  return out.join("\n");
}
