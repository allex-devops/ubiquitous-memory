from .assistant import Brief, Reply


def _warning(report) -> str:
    if report.ok:
        return ""
    bits = []
    if report.phantom:
        bits.append(f"cites passages that don't exist ({', '.join(f'[{n}]' for n in report.phantom)})")
    if report.unsupported:
        bits.append(f"{len(report.unsupported)} sentence(s) not backed by the cited passage")
    if report.uncited:
        bits.append(f"{len(report.uncited)} claim(s) with no citation")
    return "Check this answer: " + "; ".join(bits) + "."


def render_reference(r: dict) -> str:
    year = f" ({r['year']})" if r["year"] else ""
    return f"[{r['n']}] {r['title']}{year}, p.{r['page']} — {r['url']}"


def render_reply(reply: Reply) -> str:
    parts = [reply.text]
    if reply.references:
        parts.append("**Sources**\n" + "\n".join(render_reference(r) for r in reply.references))
    if reply.grounded and (warn := _warning(reply.report)):
        parts.append(warn)
    return "\n\n".join(parts)


def render_brief(brief: Brief) -> str:
    if not brief.papers:
        return brief.overview
    lines = [f"**{brief.topic}**", ""]
    if brief.overview:
        lines += [brief.overview, ""]
        if warn := _warning(brief.overview_report):
            lines += [warn, ""]
    for i, p in enumerate(brief.papers, 1):
        pages = ", ".join(str(n) for n in p.pages)
        lines += [f"[{i}] {p.title} ({p.year}), pages {pages}", f"    {p.summary}", f"    {p.url}"]
        if warn := _warning(p.report):
            lines.append(f"    {warn}")
        lines.append("")
    return "\n".join(lines).rstrip()
