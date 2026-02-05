from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from IPython.display import Markdown, display


#############################################################
# Analyze Chunks
#############################################################
def analyze_chunks_df(
    chunks: List[str],
    use_tokens: bool = True,
    tiktoken_encoding: str = "cl100k_base",
    start_index: int = 0,
    count: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - 'chunk #'  : ordinal label ('1st', '2nd', …)
      - 'text'     : the chunk text
      - 'overlap'  : maximal overlap with the NEXT chunk ('' for last row)
      - '# tokens' : token count of the chunk (tiktoken). If tiktoken isn't
                     available (or use_tokens=False), this will be None.
    """
    # ---------- helpers ----------
    def ordinal(n: int) -> str:
        if 10 <= (n % 100) <= 20:
            sfx = "th"
        else:
            sfx = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{sfx}"

    def find_char_overlap(a: str, b: str):
        m = min(len(a), len(b))
        for i in range(m, 0, -1):
            if a[-i:] == b[:i]:
                return a[-i:], i
        return "", 0

    enc = None
    if use_tokens:
        try:
            import tiktoken

            enc = tiktoken.get_encoding(tiktoken_encoding)
        except Exception:
            enc = None
            use_tokens = False  # fall back to char overlap & no token counts

    def token_overlap(a: str, b: str):
        ta, tb = enc.encode(a), enc.encode(b)
        m = min(len(ta), len(tb))
        for i in range(m, 0, -1):
            if ta[-i:] == tb[:i]:
                return enc.decode(ta[-i:]), i
        return "", 0

    def token_len(text: str):
        return len(enc.encode(text)) if enc is not None else None

    # ---------- selection bounds ----------
    n = len(chunks)
    if n == 0:
        return pd.DataFrame(columns=["chunk #", "text", "overlap", "# tokens"])

    start = max(0, start_index)
    end = n if count is None else min(n, start + max(1, count))
    if start >= n:
        return pd.DataFrame(columns=["chunk #", "text", "overlap", "# tokens"])

    # ---------- compute rows ----------
    rows = []
    for i in range(start, end):
        # overlap with next chunk (if any)
        if i + 1 < end:
            if use_tokens and enc is not None:
                ov_text, _ = token_overlap(chunks[i], chunks[i + 1])
            else:
                ov_text, _ = find_char_overlap(chunks[i], chunks[i + 1])
        else:
            ov_text = ""

        rows.append(
            {
                "chunk #": ordinal(i + 1),
                "text": chunks[i],
                "overlap": ov_text,
                "# tokens": token_len(chunks[i]) if enc is not None else None,
            }
        )

    return pd.DataFrame(rows)


#############################################################
# Preview Chunks Colored
#############################################################
def colorize_chunks_markdown(
    chunks: List[str],
    use_tokens: bool = True,
    tiktoken_encoding: str = "cl100k_base",
    overlap_color: str = "#3fc546",
    chunk_colors: tuple = ("#2382ff", "#c62828"),
    preserve_markdown: bool = True,
    add_legend: bool = True,
) -> str:
    """
    Stitch `chunks` back into the original doc using overlaps, and return
    a color-coded HTML string:
      - Each chunk's unique text is colored with alternating colors
      - Each overlap (suffix of chunk i == prefix of chunk i+1) is colored

    Render it with something like:
        from IPython.display import HTML
        HTML(colorize_chunks_markdown(chunks))

    Or in Jupyter Markdown:
        display(Markdown(colorize_chunks_markdown(chunks)))
    """
    # --- helpers -------------------------------------------------------------
    def find_char_overlap(a: str, b: str):
        m = min(len(a), len(b))
        for i in range(m, 0, -1):
            if a[-i:] == b[:i]:
                return a[-i:], i
        return "", 0

    enc = None
    if use_tokens:
        try:
            import tiktoken

            enc = tiktoken.get_encoding(tiktoken_encoding)
        except Exception:
            use_tokens = False
            enc = None

    def token_overlap(a: str, b: str):
        ta, tb = enc.encode(a), enc.encode(b)
        m = min(len(ta), len(tb))
        for i in range(m, 0, -1):
            if ta[-i:] == tb[:i]:
                return enc.decode(ta[-i:]), i
        return "", 0

    def to_html(md_text: str) -> str:
        # Convert Markdown to HTML if possible so formatting/images survive inside color spans.
        if preserve_markdown:
            try:
                import markdown as _md

                return _md.markdown(md_text, extensions=["extra", "sane_lists"])
            except Exception:
                pass  # fall through to HTML-escape
        import html

        # Escape to avoid breaking HTML; this fallback won't render Markdown, but it's safe.
        return (
            "<pre style='white-space:pre-wrap;word-wrap:break-word;margin:0'>"
            + html.escape(md_text)
            + "</pre>"
        )

    def wrap_html(html_fragment: str, color: str) -> str:
        # Use a span to color the already-rendered HTML fragment.
        return f"<span style='color:{color}'>{html_fragment}</span>"

    # --- precompute overlaps -------------------------------------------------
    n = len(chunks)
    if n == 0:
        return "<p><em>No chunks provided.</em></p>"

    overlaps = []
    for i in range(n - 1):
        if use_tokens and enc is not None:
            ov, _ = token_overlap(chunks[i], chunks[i + 1])
        else:
            ov, _ = find_char_overlap(chunks[i], chunks[i + 1])
        overlaps.append(ov or "")

    # --- build the merged, color-coded HTML ---------------------------------
    pieces = []

    # First chunk: [unique part] + [overlap with next]
    if n == 1:
        c0_html = to_html(chunks[0])
        pieces.append(wrap_html(c0_html, chunk_colors[0 % len(chunk_colors)]))
    else:
        ov0 = overlaps[0]
        cut = len(ov0)

        # unique head (chunk 0 without its trailing overlap)
        head0 = chunks[0][:-cut] if cut > 0 else chunks[0]
        head0_html = to_html(head0)
        pieces.append(wrap_html(head0_html, chunk_colors[0 % len(chunk_colors)]))

        # overlap
        if cut > 0:
            ov0_html = to_html(ov0)
            pieces.append(wrap_html(ov0_html, overlap_color))

    # Middle chunks
    for i in range(1, n - 1):
        prev_ov = overlaps[i - 1]
        next_ov = overlaps[i]
        left = len(prev_ov)
        right = len(next_ov)

        # middle (skip the prefix that overlapped with prev, and the suffix that overlaps with next)
        mid = chunks[i][left : len(chunks[i]) - right if right > 0 else None]
        if mid:
            mid_html = to_html(mid)
            pieces.append(wrap_html(mid_html, chunk_colors[i % len(chunk_colors)]))

        # next overlap
        if next_ov:
            next_ov_html = to_html(next_ov)
            pieces.append(wrap_html(next_ov_html, overlap_color))

    # Last chunk: skip its prefix that overlapped with previous, then color the rest
    if n >= 2:
        last_prefix = overlaps[-1]
        lp = len(last_prefix)
        tail = chunks[-1][lp:] if lp > 0 else chunks[-1]
        if tail:
            tail_html = to_html(tail)
            pieces.append(wrap_html(tail_html, chunk_colors[(n - 1) % len(chunk_colors)]))

    body_html = "".join(pieces)

    legend = ""
    if add_legend:
        swatches = [
            ("Chunk 1", chunk_colors[0 % len(chunk_colors)]),
            ("Overlap", overlap_color),
            ("Chunk 2", chunk_colors[1 % len(chunk_colors)]),
        ]
        legend_items = "".join(
            f"<li><span style='display:inline-block;width:0.9em;height:0.9em;"
            f"background:{c};margin-right:0.5em;border-radius:3px;"
            f"vertical-align:middle'></span>{name}</li>"
            for name, c in swatches
        )
        legend = f"""
        <div style="font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial">
          <strong>Legend:</strong>
          <ul style="margin:0.4em 0 1em 1.2em">{legend_items}</ul>
        </div>
        """

    # minimal container styling so it looks nice in notebooks/renderers
    container_css = """
        max-width: 900px;
        line-height: 1.6;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        word-break: break-word;
    """

    return f"""
<div style="{container_css}">
  {legend}
  {body_html}
</div>
""".strip()


