from __future__ import annotations

import csv
import html
import io
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


# ============================================================
# App configuration
# ============================================================

API_URL = os.getenv("NER_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Анализ сущностей",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette inspired by Brand Analytics' published visual identity:
# vivid green + blue + dark graphite, without reproducing their logo.
BRAND = {
    "green": "#8BC53F",
    "green_dark": "#659C1D",
    "blue": "#4E78C4",
    "blue_dark": "#355C9F",
    "graphite": "#313743",
    "graphite_2": "#4B5361",
    "background": "#F4F6F8",
    "line": "#E1E5EA",
}

LABEL_META = {
    "ORG": {
        "title": "Организация",
        "short": "ORG",
        "css": "org",
    },
    "NAME": {
        "title": "Персона",
        "short": "NAME",
        "css": "name",
    },
    "GEO": {
        "title": "География",
        "short": "GEO",
        "css": "geo",
    },
}

LABEL_TITLES = {k: v["title"] for k, v in LABEL_META.items()}
TITLE_TO_LABEL = {v["title"]: k for k, v in LABEL_META.items()}

EXAMPLES = {
    "Пример на латинице": (
        "Toshkent shahrida O‘zbekiston Respublikasi Markaziy banki "
        "vakillari bilan uchrashuv bo‘lib o‘tdi."
    ),
    "Пример на кириллице": (
        "Тошкент шаҳрида Ўзбекистон Республикаси Марказий банки "
        "вакиллари билан учрашув бўлиб ўтди."
    ),
    "Пример со смешанным текстом": (
        "Toshkentdagi tadbirda Samsung Uzbekistan vakillari va "
        "Akmal Xolmatov иштирок этди."
    ),
}


# ============================================================
# Styling
# ============================================================

st.markdown(
    f"""
<style>
:root {{
    --ba-green: {BRAND["green"]};
    --ba-green-dark: {BRAND["green_dark"]};
    --ba-blue: {BRAND["blue"]};
    --ba-blue-dark: {BRAND["blue_dark"]};
    --graphite: {BRAND["graphite"]};
    --graphite-2: {BRAND["graphite_2"]};
    --bg: {BRAND["background"]};
    --line: {BRAND["line"]};

    --org-bg: #EAF1FF;
    --org-accent: #4E78C4;
    --org-text: #355C9F;

    --name-bg: #ECF8DF;
    --name-accent: #8BC53F;
    --name-text: #527F1B;

    --geo-bg: #FFF1D9;
    --geo-accent: #E9A33A;
    --geo-text: #9A6414;
}}

.stApp {{
    background:
        radial-gradient(circle at 82% -14%, rgba(139,197,63,.10), transparent 32rem),
        radial-gradient(circle at 7% -8%, rgba(78,120,196,.10), transparent 30rem),
        var(--bg);
}}

.block-container {{
    max-width: 1200px;
    padding-top: 2.1rem;
    padding-bottom: 4rem;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: #2F3540;
    border-right: none;
}}

section[data-testid="stSidebar"] * {{
    color: #F4F6F8;
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button {{
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.10);
    color: #F7F8FA;
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {{
    border-color: var(--ba-green);
    color: #FFFFFF;
}}

.sidebar-brand {{
    margin: .35rem 0 1.4rem;
}}

.sidebar-brand-mark {{
    display: inline-flex;
    width: 33px;
    height: 33px;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--ba-green) 0 48%, var(--ba-blue) 48% 100%);
    color: white;
    font-size: 1rem;
    font-weight: 800;
}}

.sidebar-brand-title {{
    margin-top: .7rem;
    font-weight: 720;
    font-size: .96rem;
    color: #FFFFFF;
}}

.sidebar-muted {{
    color: #AFB6C1 !important;
    font-size: .78rem;
    line-height: 1.45;
}}

/* Hero */
.hero {{
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
    padding: 1.85rem 2rem;
    border-radius: 22px;
    background: #303742;
    color: white;
    box-shadow: 0 16px 42px rgba(25,31,40,.10);
}}

.hero::after {{
    content: "";
    position: absolute;
    width: 290px;
    height: 290px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,197,63,.32), rgba(139,197,63,0));
}}

.hero-kicker {{
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    margin-bottom: .75rem;
    color: #C8D0D9;
    font-size: .78rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

.hero-kicker-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--ba-green);
}}

.hero-title {{
    position: relative;
    z-index: 1;
    margin: 0;
    max-width: 760px;
    color: #FFFFFF;
    font-size: clamp(2rem, 5vw, 3.4rem);
    line-height: 1.03;
    letter-spacing: -.045em;
    font-weight: 780;
}}

.hero-subtitle {{
    position: relative;
    z-index: 1;
    margin-top: .75rem;
    max-width: 760px;
    color: #C9D0D8;
    font-size: 1rem;
    line-height: 1.6;
}}

.service-pill {{
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: .42rem;
    margin-top: 1rem;
    padding: .33rem .60rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.11);
    background: rgba(255,255,255,.07);
    color: #DCE2E8;
    font-size: .77rem;
}}

.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}

.status-ok {{
    background: var(--ba-green);
    box-shadow: 0 0 0 3px rgba(139,197,63,.15);
}}

.status-bad {{
    background: #ED6A5A;
    box-shadow: 0 0 0 3px rgba(237,106,90,.14);
}}

/* Common cards */
.surface {{
    background: rgba(255,255,255,.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 2px 5px rgba(20,30,45,.025);
}}

.section-title {{
    color: var(--graphite);
    font-size: 1.28rem;
    font-weight: 740;
    margin: 1.5rem 0 .2rem;
}}

.section-hint {{
    color: #78818E;
    font-size: .86rem;
    margin-bottom: .8rem;
}}

/* Metrics as user summary, not model metrics */
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .7rem;
    margin: .7rem 0 1rem;
}}

.summary-card {{
    padding: .85rem .95rem;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
}}

.summary-value {{
    color: var(--graphite);
    font-size: 1.55rem;
    line-height: 1.1;
    font-weight: 780;
}}

.summary-label {{
    margin-top: .28rem;
    color: #818895;
    font-size: .78rem;
}}

.summary-card.total {{
    border-top: 3px solid var(--graphite);
}}

.summary-card.org {{
    border-top: 3px solid var(--org-accent);
}}

.summary-card.name {{
    border-top: 3px solid var(--name-accent);
}}

.summary-card.geo {{
    border-top: 3px solid var(--geo-accent);
}}

/* Textarea */
div[data-testid="stTextArea"] textarea {{
    min-height: 210px !important;
    border-radius: 15px !important;
    border: 1px solid #D9DEE5 !important;
    background: #FFFFFF !important;
    color: var(--graphite) !important;
    font-size: 1rem !important;
    line-height: 1.58 !important;
    padding: 1rem !important;
}}

div[data-testid="stTextArea"] textarea:focus {{
    border-color: var(--ba-blue) !important;
    box-shadow: 0 0 0 3px rgba(78,120,196,.10) !important;
}}

/* Buttons */
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {{
    min-height: 2.55rem;
    border-radius: 11px;
    font-weight: 650;
}}

div[data-testid="stFormSubmitButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"] {{
    background: var(--ba-green);
    border-color: var(--ba-green);
    color: #26311C;
}}

div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover {{
    background: #95D046;
    border-color: #95D046;
}}

/* Highlight */
.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin: .55rem 0 .8rem;
}}

.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: .38rem;
    padding: .28rem .52rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: #FFF;
    color: #5D6572;
    font-size: .78rem;
}}

.legend-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
}}

.highlight-box {{
    padding: 1.2rem 1.25rem;
    border: 1px solid var(--line);
    border-radius: 17px;
    background: #FFFFFF;
    color: #2F3540;
    font-size: 1.01rem;
    line-height: 2.2;
    overflow-wrap: anywhere;
}}

.entity {{
    display: inline;
    margin: 0 .03rem;
    padding: .11rem .27rem .15rem;
    border-radius: 6px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}}

.entity-label {{
    display: inline-block;
    margin-left: .28rem;
    padding: .045rem .28rem;
    border-radius: 999px;
    font-size: .60rem;
    line-height: 1.4;
    font-weight: 800;
    letter-spacing: .025em;
    vertical-align: .12em;
}}

.entity-confidence {{
    opacity: .72;
    margin-left: .20rem;
}}

.entity-org {{
    background: var(--org-bg);
    box-shadow: inset 0 -2px 0 var(--org-accent);
}}
.entity-org .entity-label {{
    background: #DDE8FF;
    color: var(--org-text);
}}

.entity-name {{
    background: var(--name-bg);
    box-shadow: inset 0 -2px 0 var(--name-accent);
}}
.entity-name .entity-label {{
    background: #E0F2CA;
    color: var(--name-text);
}}

.entity-geo {{
    background: var(--geo-bg);
    box-shadow: inset 0 -2px 0 var(--geo-accent);
}}
.entity-geo .entity-label {{
    background: #FFE5BA;
    color: var(--geo-text);
}}

/* Entity table */
.table-wrap {{
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 15px;
    background: #FFF;
}}

.entity-table {{
    width: 100%;
    min-width: 690px;
    border-collapse: collapse;
}}

.entity-table th {{
    padding: .78rem .88rem;
    text-align: left;
    background: #F8F9FA;
    border-bottom: 1px solid #E8EBEF;
    color: #838B97;
    font-size: .71rem;
    font-weight: 720;
    text-transform: uppercase;
    letter-spacing: .035em;
}}

.entity-table td {{
    padding: .84rem .88rem;
    border-bottom: 1px solid #EEF0F2;
    color: #39404B;
    font-size: .89rem;
    vertical-align: middle;
}}

.entity-table tr:last-child td {{
    border-bottom: none;
}}

.entity-value {{
    color: #252B34;
    font-weight: 670;
}}

.type-badge {{
    display: inline-flex;
    padding: .26rem .47rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 680;
    white-space: nowrap;
}}

.badge-org {{
    background: var(--org-bg);
    color: var(--org-text);
}}
.badge-name {{
    background: var(--name-bg);
    color: var(--name-text);
}}
.badge-geo {{
    background: var(--geo-bg);
    color: var(--geo-text);
}}

.context {{
    color: #747D89;
    line-height: 1.4;
}}

.count-badge {{
    display: inline-flex;
    min-width: 28px;
    justify-content: center;
    padding: .22rem .42rem;
    border-radius: 999px;
    background: #F0F2F4;
    color: #56606D;
    font-weight: 650;
}}

/* Empty/error */
.empty-state {{
    padding: 1.8rem 1rem;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #FFF;
    color: #737C88;
    text-align: center;
}}

/* Tabs and expanders */
div[data-testid="stTabs"] button {{
    font-weight: 650;
}}

div[data-testid="stExpander"] {{
    border: 1px solid var(--line);
    border-radius: 14px;
    background: rgba(255,255,255,.90);
}}

/* Batch */
.batch-doc {{
    padding: .75rem .85rem;
    border-radius: 12px;
    background: #FFF;
    border: 1px solid var(--line);
    margin-bottom: .5rem;
}}

/* Responsive */
@media (max-width: 760px) {{
    .block-container {{
        padding-left: .9rem;
        padding-right: .9rem;
    }}

    .hero {{
        padding: 1.5rem 1.25rem;
        border-radius: 18px;
    }}

    .summary-grid {{
        grid-template-columns: repeat(2, minmax(0,1fr));
    }}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# API helpers
# ============================================================

def api_get(path: str, timeout: int = 10) -> dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_predict(text: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# State helpers
# ============================================================

def reset_current_result() -> None:
    st.session_state.current_text = None
    st.session_state.current_entities = []
    st.session_state.raw_result = None
    st.session_state.last_error = None


def set_text(value: str) -> None:
    st.session_state.analysis_text = value
    reset_current_result()


def restore_history(index: int) -> None:
    item = st.session_state.history[index]
    st.session_state.analysis_text = item["text"]
    st.session_state.current_text = item["text"]
    st.session_state.current_entities = deepcopy(item["entities"])
    st.session_state.raw_result = deepcopy(item.get("raw_result"))
    st.session_state.last_error = None


def add_history(text: str, entities: list[dict[str, Any]], raw_result: dict[str, Any]) -> None:
    item = {
        "created_at": datetime.now().strftime("%H:%M"),
        "text": text,
        "entities": deepcopy(entities),
        "raw_result": deepcopy(raw_result),
    }

    # Avoid repeated identical items at the top.
    if st.session_state.history:
        top = st.session_state.history[0]
        if top["text"] == text:
            st.session_state.history[0] = item
            return

    st.session_state.history.insert(0, item)
    st.session_state.history = st.session_state.history[:10]


# ============================================================
# Rendering helpers
# ============================================================

def confidence_value(entity: dict[str, Any]) -> float | None:
    value = entity.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def entity_text(text: str, entity: dict[str, Any]) -> str:
    if entity.get("text"):
        return str(entity["text"])
    return text[int(entity["start"]):int(entity["end"])]


def visible_entities(
    entities: list[dict[str, Any]],
    selected_labels: list[str],
    min_confidence: float,
) -> list[dict[str, Any]]:
    result = []

    for entity in entities:
        if entity.get("label") not in selected_labels:
            continue

        conf = confidence_value(entity)

        # Manual entities do not have a model confidence and should stay visible.
        if conf is not None and conf < min_confidence:
            continue

        result.append(entity)

    return sorted(result, key=lambda e: (int(e["start"]), int(e["end"])))


def summary_html(entities: list[dict[str, Any]]) -> str:
    counts = Counter(str(e["label"]) for e in entities)

    return f"""
<div class="summary-grid">
    <div class="summary-card total">
        <div class="summary-value">{len(entities)}</div>
        <div class="summary-label">Всего сущностей</div>
    </div>
    <div class="summary-card org">
        <div class="summary-value">{counts.get("ORG", 0)}</div>
        <div class="summary-label">Организации</div>
    </div>
    <div class="summary-card name">
        <div class="summary-value">{counts.get("NAME", 0)}</div>
        <div class="summary-label">Персоны</div>
    </div>
    <div class="summary-card geo">
        <div class="summary-value">{counts.get("GEO", 0)}</div>
        <div class="summary-label">География</div>
    </div>
</div>
"""


def legend_html() -> str:
    data = [
        ("#4E78C4", "Организация"),
        ("#8BC53F", "Персона"),
        ("#E9A33A", "География"),
    ]

    return (
        '<div class="legend">'
        + "".join(
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{color}"></span>'
            f'{title}</span>'
            for color, title in data
        )
        + "</div>"
    )


def highlighted_html(
    text: str,
    entities: list[dict[str, Any]],
    show_confidence: bool,
) -> str:
    parts: list[str] = []
    cursor = 0

    for entity in sorted(
        entities,
        key=lambda e: (int(e["start"]), int(e["end"])),
    ):
        start = int(entity["start"])
        end = int(entity["end"])
        label = str(entity["label"])

        if start < cursor or not (0 <= start < end <= len(text)):
            continue

        meta = LABEL_META.get(label, {"css": "org", "short": label})
        parts.append(html.escape(text[cursor:start]))

        conf = confidence_value(entity)
        conf_html = ""
        if show_confidence:
            conf_html = (
                '<span class="entity-confidence">'
                + ("manual" if conf is None else f"{conf:.2f}")
                + "</span>"
            )

        parts.append(
            f'<span class="entity entity-{meta["css"]}">'
            f'{html.escape(text[start:end])}'
            f'<span class="entity-label">{html.escape(meta["short"])}{conf_html}</span>'
            f'</span>'
        )
        cursor = end

    parts.append(html.escape(text[cursor:]))

    return (
        '<div class="highlight-box">'
        + "".join(parts).replace("\n", "<br>")
        + "</div>"
    )


def context_for_span(text: str, start: int, end: int, radius: int = 36) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return prefix + text[left:right].replace("\n", " ") + suffix


def entity_table_html(
    text: str,
    entities: list[dict[str, Any]],
    group_duplicates: bool,
    show_confidence: bool,
) -> str:
    rows: list[str] = []

    if group_duplicates:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for entity in entities:
            value = entity_text(text, entity).strip()
            key = (value.casefold(), str(entity["label"]))
            groups[key].append(entity)

        ordered_groups = sorted(
            groups.values(),
            key=lambda items: (
                int(items[0]["start"]),
                str(items[0]["label"]),
            ),
        )

        for items in ordered_groups:
            first = items[0]
            label = str(first["label"])
            meta = LABEL_META.get(label, {"title": label, "css": "org"})
            value = html.escape(entity_text(text, first))
            contexts = [
                context_for_span(text, int(e["start"]), int(e["end"]))
                for e in items[:2]
            ]
            context = html.escape(" / ".join(contexts))

            conf_values = [
                confidence_value(e)
                for e in items
                if confidence_value(e) is not None
            ]
            conf_cell = ""
            if show_confidence:
                conf_cell = (
                    f"<td>{sum(conf_values)/len(conf_values):.2f}</td>"
                    if conf_values
                    else "<td>вручную</td>"
                )

            rows.append(
                "<tr>"
                f'<td><span class="entity-value">{value}</span></td>'
                f'<td><span class="type-badge badge-{meta["css"]}">{html.escape(meta["title"])}</span></td>'
                f'<td><span class="count-badge">{len(items)}</span></td>'
                f'<td><span class="context">{context}</span></td>'
                f"{conf_cell}"
                "</tr>"
            )

        confidence_header = "<th>Confidence</th>" if show_confidence else ""

        header = (
            "<tr><th>Сущность</th><th>Тип</th>"
            "<th>Упоминаний</th><th>Контекст</th>"
            f"{confidence_header}</tr>"
        )
    else:
        for entity in entities:
            label = str(entity["label"])
            meta = LABEL_META.get(label, {"title": label, "css": "org"})
            value = html.escape(entity_text(text, entity))
            context = html.escape(
                context_for_span(text, int(entity["start"]), int(entity["end"]))
            )

            conf = confidence_value(entity)
            conf_cell = ""
            if show_confidence:
                conf_cell = (
                    "<td>вручную</td>"
                    if conf is None
                    else f"<td>{conf:.2f}</td>"
                )

            rows.append(
                "<tr>"
                f'<td><span class="entity-value">{value}</span></td>'
                f'<td><span class="type-badge badge-{meta["css"]}">{html.escape(meta["title"])}</span></td>'
                f'<td><span class="context">{context}</span></td>'
                f"{conf_cell}"
                "</tr>"
            )

        confidence_header = "<th>Confidence</th>" if show_confidence else ""
        header = (
            "<tr><th>Сущность</th><th>Тип</th><th>Контекст</th>"
            f"{confidence_header}</tr>"
        )

    return (
        '<div class="table-wrap">'
        '<table class="entity-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def make_export_rows(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "text": entity_text(text, e),
            "label": e["label"],
            "start": int(e["start"]),
            "end": int(e["end"]),
            "confidence": confidence_value(e),
            "source": e.get("source", "model"),
        }
        for e in sorted(entities, key=lambda x: (int(x["start"]), int(x["end"])))
    ]


def export_json(text: str, entities: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "text": text,
            "entities": make_export_rows(text, entities),
        },
        ensure_ascii=False,
        indent=2,
    )


def export_jsonl(text: str, entities: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "text": text,
            "entities": make_export_rows(text, entities),
        },
        ensure_ascii=False,
    ) + "\n"


def export_csv(text: str, entities: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    rows = make_export_rows(text, entities)
    fieldnames = ["text", "label", "start", "end", "confidence", "source"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if not needle:
        return positions

    start_from = 0

    while True:
        pos = text.find(needle, start_from)
        if pos < 0:
            break
        positions.append((pos, pos + len(needle)))
        start_from = pos + max(1, len(needle))

    return positions


# ============================================================
# Batch helpers
# ============================================================

def parse_batch_file(uploaded_file) -> tuple[list[dict[str, Any]], str | None]:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".txt"):
        text = raw.decode("utf-8-sig")
        docs = [
            {"id": i + 1, "text": line.strip()}
            for i, line in enumerate(text.splitlines())
            if line.strip()
        ]
        return docs, None

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty:
            return [], "CSV пуст."

        candidate_columns = [
            c for c in df.columns
            if str(c).lower() in {"text", "message", "content", "sentence"}
        ]

        text_column = candidate_columns[0] if candidate_columns else None

        if text_column is None:
            object_columns = list(df.select_dtypes(include="object").columns)
            if not object_columns:
                return [], (
                    "В CSV не найдена текстовая колонка. "
                    "Используйте колонку text, message, content или sentence."
                )
            text_column = object_columns[0]

        docs = []
        for idx, row in df.iterrows():
            value = row.get(text_column)
            if pd.isna(value) or not str(value).strip():
                continue
            docs.append({
                "id": row.get("id", idx + 1),
                "text": str(value),
            })

        return docs, None

    if name.endswith(".jsonl"):
        decoded = raw.decode("utf-8-sig")
        docs = []

        for i, line in enumerate(decoded.splitlines(), start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                return [], f"Ошибка JSONL в строке {i}: {exc}"

            if isinstance(obj, str):
                docs.append({"id": i, "text": obj})
                continue

            if not isinstance(obj, dict) or not str(obj.get("text", "")).strip():
                return [], f"В строке {i} JSONL нет непустого поля text."

            docs.append({
                "id": obj.get("id", i),
                "text": str(obj["text"]),
            })

        return docs, None

    return [], "Поддерживаются .txt, .csv и .jsonl."


def batch_csv(results: list[dict[str, Any]]) -> str:
    rows = []
    for result in results:
        counts = Counter(e["label"] for e in result["entities"])
        rows.append({
            "id": result["id"],
            "text": result["text"],
            "entities_count": len(result["entities"]),
            "org_count": counts.get("ORG", 0),
            "name_count": counts.get("NAME", 0),
            "geo_count": counts.get("GEO", 0),
            "entities_json": json.dumps(
                result["entities"],
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(rows).to_csv(index=False)


def batch_jsonl(results: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(result, ensure_ascii=False) + "\n"
        for result in results
    )


# ============================================================
# Initialize state
# ============================================================

defaults = {
    "analysis_text": "",
    "current_text": None,
    "current_entities": [],
    "raw_result": None,
    "last_error": None,
    "history": [],
    "batch_results": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = deepcopy(value)


# ============================================================
# Service status
# ============================================================

service_ok = False
service_device = None
model_info_cache = None

try:
    health = api_get("/health", timeout=4)
    service_ok = bool(health.get("model_loaded", True))
    service_device = health.get("device")
except Exception:
    pass


# ============================================================
# Sidebar — history
# ============================================================

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="sidebar-brand-mark">A</div>
    <div class="sidebar-brand-title">Анализ сущностей</div>
    <div class="sidebar-muted">
        Поиск людей, организаций и географических объектов
        в узбекских текстах.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### История")

    if not st.session_state.history:
        st.caption("Здесь появятся последние запросы.")
    else:
        for i, item in enumerate(st.session_state.history):
            preview = item["text"].replace("\n", " ").strip()
            if len(preview) > 45:
                preview = preview[:45] + "…"

            st.button(
                f'{item["created_at"]} · {preview}',
                key=f"history_{i}",
                use_container_width=True,
                on_click=restore_history,
                args=(i,),
            )

        if st.button("Очистить историю", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ============================================================
# Hero
# ============================================================

status_text = "Сервис готов" if service_ok else "Backend недоступен"
status_class = "status-ok" if service_ok else "status-bad"

st.markdown(
    f"""
<div class="hero">
    <div class="hero-kicker">
        <span class="hero-kicker-dot"></span>
        AI-анализ узбекского текста
    </div>
    <h1 class="hero-title">Анализ сущностей в тексте</h1>
    <div class="hero-subtitle">
        Автоматически находите людей, организации и географические объекты.
        Проверьте результат, исправьте разметку и выгрузите данные в удобном формате.
    </div>
    <div class="service-pill">
        <span class="status-dot {status_class}"></span>
        {status_text}
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Main navigation
# ============================================================

analysis_tab, batch_tab = st.tabs(
    ["Анализ одного текста", "Пакетная обработка"]
)


# ============================================================
# Single document analysis
# ============================================================

with analysis_tab:
    st.markdown(
        '<div class="section-title">Текст для анализа</div>'
        '<div class="section-hint">'
        'Введите текст вручную, используйте пример или загрузите .txt файл.'
        '</div>',
        unsafe_allow_html=True,
    )

    example_cols = st.columns(4)

    for col, (title, value) in zip(example_cols[:3], EXAMPLES.items()):
        col.button(
            title,
            key=f"example_{title}",
            use_container_width=True,
            on_click=set_text,
            args=(value,),
        )

    example_cols[3].button(
        "Очистить",
        key="clear_single_text",
        use_container_width=True,
        on_click=set_text,
        args=("",),
    )

    uploaded_txt = st.file_uploader(
        "Загрузить TXT",
        type=["txt"],
        key="single_txt_upload",
        help="Файл должен быть в UTF-8. Он будет загружен в поле ввода.",
    )

    if uploaded_txt is not None:
        try:
            decoded_txt = uploaded_txt.getvalue().decode("utf-8-sig")
            if st.button(
                f"Использовать текст из {uploaded_txt.name}",
                key="use_uploaded_txt",
            ):
                set_text(decoded_txt)
                st.rerun()
        except UnicodeDecodeError:
            st.error("Не удалось прочитать файл как UTF-8.")

    with st.form("single_analysis_form"):
        text = st.text_area(
            "Текст",
            key="analysis_text",
            label_visibility="collapsed",
            placeholder=(
                "Masalan: Toshkent shahrida O‘zbekiston Respublikasi "
                "Markaziy banki vakillari..."
            ),
        )

        submitted = st.form_submit_button(
            "Найти сущности",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not text.strip():
            reset_current_result()
            st.session_state.last_error = "Введите текст для анализа."
        elif not service_ok:
            reset_current_result()
            st.session_state.last_error = (
                "Backend недоступен. Проверьте, что FastAPI запущен."
            )
        else:
            try:
                with st.spinner("Анализируем текст…"):
                    result = api_predict(text)

                entities = []
                for entity in result.get("entities", []):
                    entity = dict(entity)
                    entity.setdefault("source", "model")
                    entities.append(entity)

                st.session_state.current_text = text
                st.session_state.current_entities = entities
                st.session_state.raw_result = result
                st.session_state.last_error = None

                add_history(text, entities, result)

            except requests.HTTPError as exc:
                reset_current_result()
                st.session_state.last_error = (
                    f"Сервис вернул ошибку: {exc.response.text}"
                )
            except Exception as exc:
                reset_current_result()
                st.session_state.last_error = (
                    f"Не удалось выполнить анализ: {exc}"
                )

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    current_text = st.session_state.current_text
    current_entities = st.session_state.current_entities

    if current_text is not None:
        # Advanced display controls.
        with st.expander("Настройки отображения", expanded=False):
            selected_titles = st.multiselect(
                "Какие сущности показывать",
                options=[LABEL_TITLES["ORG"], LABEL_TITLES["NAME"], LABEL_TITLES["GEO"]],
                default=[LABEL_TITLES["ORG"], LABEL_TITLES["NAME"], LABEL_TITLES["GEO"]],
                key="label_filter_titles",
            )

            selected_labels = [
                TITLE_TO_LABEL[title]
                for title in selected_titles
            ]

            group_duplicates = st.toggle(
                "Группировать одинаковые сущности",
                value=True,
                key="group_duplicates",
            )

            show_confidence = st.toggle(
                "Показывать confidence",
                value=False,
                key="show_confidence",
            )

            min_confidence = st.slider(
                "Дополнительный порог confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="frontend_min_confidence",
                help=(
                    "Фильтр применяется поверх порогов backend. "
                    "Он может скрыть менее уверенные сущности, но не вернуть "
                    "кандидаты, уже отброшенные моделью."
                ),
            )

        display_entities = visible_entities(
            current_entities,
            selected_labels=selected_labels,
            min_confidence=min_confidence,
        )

        st.markdown(
            '<div class="section-title">Результат</div>'
            '<div class="section-hint">'
            'Сводка относится к сущностям, которые сейчас отображаются.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            summary_html(display_entities),
            unsafe_allow_html=True,
        )

        if display_entities:
            st.markdown(
                legend_html(),
                unsafe_allow_html=True,
            )

            st.markdown(
                highlighted_html(
                    current_text,
                    display_entities,
                    show_confidence=show_confidence,
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-title">Найденные сущности</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                entity_table_html(
                    current_text,
                    display_entities,
                    group_duplicates=group_duplicates,
                    show_confidence=show_confidence,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">'
                'По текущим фильтрам сущности не найдены.'
                '</div>',
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # Manual review / correction
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Проверка разметки</div>'
            '<div class="section-hint">'
            'Измените тип сущности, удалите ошибочную или добавьте пропущенную.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Исправить найденные сущности", expanded=False):
            sorted_entities = sorted(
                current_entities,
                key=lambda e: (int(e["start"]), int(e["end"])),
            )

            editor_rows = []

            for entity in sorted_entities:
                editor_rows.append({
                    "Оставить": True,
                    "Сущность": entity_text(current_text, entity),
                    "Тип": LABEL_TITLES.get(
                        str(entity["label"]),
                        str(entity["label"]),
                    ),
                    "Позиция": f'{int(entity["start"])}–{int(entity["end"])}',
                    "Confidence": (
                        "вручную"
                        if confidence_value(entity) is None
                        else f'{confidence_value(entity):.3f}'
                    ),
                })

            if editor_rows:
                edited_df = st.data_editor(
                    pd.DataFrame(editor_rows),
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Сущность", "Позиция", "Confidence"],
                    column_config={
                        "Оставить": st.column_config.CheckboxColumn(
                            "Оставить",
                            help="Снимите галочку, чтобы удалить сущность.",
                        ),
                        "Тип": st.column_config.SelectboxColumn(
                            "Тип",
                            options=[
                                LABEL_TITLES["ORG"],
                                LABEL_TITLES["NAME"],
                                LABEL_TITLES["GEO"],
                            ],
                            required=True,
                        ),
                    },
                    key="entity_editor",
                )

                if st.button(
                    "Применить правки",
                    key="apply_entity_edits",
                    type="primary",
                ):
                    updated_entities = []

                    for original, (_, edited_row) in zip(
                        sorted_entities,
                        edited_df.iterrows(),
                    ):
                        if not bool(edited_row["Оставить"]):
                            continue

                        updated = dict(original)
                        updated["label"] = TITLE_TO_LABEL[str(edited_row["Тип"])]
                        if updated["label"] != original["label"]:
                            updated["source"] = "manual_edit"

                        updated_entities.append(updated)

                    st.session_state.current_entities = updated_entities
                    st.success("Правки применены.")
                    st.rerun()
            else:
                st.caption("Сейчас нет сущностей для редактирования.")

        with st.expander("Добавить пропущенную сущность", expanded=False):
            mention = st.text_input(
                "Точный текст сущности",
                key="manual_mention",
                placeholder="Например: Toshkent",
            )

            occurrences = find_occurrences(current_text, mention) if mention else []

            if mention and not occurrences:
                st.warning("Такой фрагмент не найден в исходном тексте.")

            occurrence_options = []

            for index, (start, end) in enumerate(occurrences):
                ctx = context_for_span(current_text, start, end, radius=28)
                occurrence_options.append(
                    f"{index + 1}. позиция {start}–{end}: {ctx}"
                )

            selected_occurrence = None

            if occurrence_options:
                selected_occurrence = st.selectbox(
                    "Какое упоминание добавить",
                    occurrence_options,
                    key="manual_occurrence",
                )

            manual_label_title = st.selectbox(
                "Тип сущности",
                [
                    LABEL_TITLES["ORG"],
                    LABEL_TITLES["NAME"],
                    LABEL_TITLES["GEO"],
                ],
                key="manual_label",
            )

            if st.button(
                "Добавить сущность",
                key="add_manual_entity",
                disabled=not bool(occurrences),
            ):
                selected_index = occurrence_options.index(selected_occurrence)
                start, end = occurrences[selected_index]
                label = TITLE_TO_LABEL[manual_label_title]

                duplicate = any(
                    int(e["start"]) == start
                    and int(e["end"]) == end
                    and str(e["label"]) == label
                    for e in st.session_state.current_entities
                )

                if duplicate:
                    st.info("Такая сущность уже есть в разметке.")
                else:
                    st.session_state.current_entities.append({
                        "text": current_text[start:end],
                        "label": label,
                        "start": start,
                        "end": end,
                        "confidence": None,
                        "source": "manual_add",
                    })
                    st.success("Сущность добавлена.")
                    st.rerun()

        # ----------------------------------------------------
        # Export
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Экспорт</div>'
            '<div class="section-hint">'
            'Выгружается текущая исправленная разметка без визуальных фильтров.'
            '</div>',
            unsafe_allow_html=True,
        )

        export_cols = st.columns(3)

        export_cols[0].download_button(
            "Скачать CSV",
            data=export_csv(current_text, current_entities),
            file_name="entities.csv",
            mime="text/csv",
            use_container_width=True,
        )

        export_cols[1].download_button(
            "Скачать JSON",
            data=export_json(current_text, current_entities),
            file_name="entities.json",
            mime="application/json",
            use_container_width=True,
        )

        export_cols[2].download_button(
            "Скачать JSONL",
            data=export_jsonl(current_text, current_entities),
            file_name="entities.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        with st.expander("Техническая информация", expanded=False):
            raw_result = st.session_state.raw_result or {}

            latency = raw_result.get("latency_ms")
            if latency is not None:
                st.write(f"**Время inference:** {float(latency):.1f} ms")

            st.write(
                f"**Backend:** `{API_URL}` · "
                f"**Device:** `{service_device or 'unknown'}`"
            )

            try:
                info = api_get("/model-info", timeout=8)
                st.write("**Параметры модели и сервиса**")
                st.json(info)
            except Exception:
                st.caption("Не удалось получить /model-info.")

            st.write("**Текущая разметка**")
            st.json({
                "text": current_text,
                "entities": make_export_rows(current_text, current_entities),
            })


# ============================================================
# Batch processing
# ============================================================

with batch_tab:
    st.markdown(
        '<div class="section-title">Пакетная обработка</div>'
        '<div class="section-hint">'
        'Загрузите TXT, CSV или JSONL и обработайте несколько документов за один запуск.'
        '</div>',
        unsafe_allow_html=True,
    )

    batch_upload = st.file_uploader(
        "Файл с документами",
        type=["txt", "csv", "jsonl"],
        key="batch_upload",
        help=(
            "TXT: один документ на строку. "
            "CSV: предпочтительно колонка text. "
            "JSONL: объект с полем text на каждой строке."
        ),
    )

    parsed_docs: list[dict[str, Any]] = []
    parse_error = None

    if batch_upload is not None:
        parsed_docs, parse_error = parse_batch_file(batch_upload)

        if parse_error:
            st.error(parse_error)
        else:
            st.success(f"Готово к обработке: {len(parsed_docs)} документов")

            if parsed_docs:
                preview = pd.DataFrame(parsed_docs[:5])
                preview["text"] = preview["text"].astype(str).str.slice(0, 180)
                st.dataframe(
                    preview,
                    hide_index=True,
                    use_container_width=True,
                )

    if st.button(
        "Обработать пакет",
        key="run_batch",
        type="primary",
        use_container_width=True,
        disabled=(
            not service_ok
            or not parsed_docs
            or parse_error is not None
        ),
    ):
        results = []
        progress = st.progress(0, text="Начинаем обработку…")
        status = st.empty()

        total = len(parsed_docs)

        for index, doc in enumerate(parsed_docs, start=1):
            status.write(f"Документ {index} из {total}")

            try:
                prediction = api_predict(str(doc["text"]))
                entities = []

                for entity in prediction.get("entities", []):
                    entity = dict(entity)
                    entity.setdefault("source", "model")
                    entities.append(entity)

                results.append({
                    "id": doc["id"],
                    "text": str(doc["text"]),
                    "entities": entities,
                    "latency_ms": prediction.get("latency_ms"),
                })
            except Exception as exc:
                results.append({
                    "id": doc["id"],
                    "text": str(doc["text"]),
                    "entities": [],
                    "error": str(exc),
                })

            progress.progress(
                index / total,
                text=f"Обработано {index} из {total}",
            )

        status.empty()
        st.session_state.batch_results = results
        st.success("Пакет обработан.")

    batch_results = st.session_state.batch_results

    if batch_results:
        all_entities = [
            entity
            for result in batch_results
            for entity in result.get("entities", [])
        ]
        counts = Counter(e["label"] for e in all_entities)

        st.markdown(
            f"""
<div class="summary-grid">
    <div class="summary-card total">
        <div class="summary-value">{len(batch_results)}</div>
        <div class="summary-label">Документов</div>
    </div>
    <div class="summary-card org">
        <div class="summary-value">{len(all_entities)}</div>
        <div class="summary-label">Всего сущностей</div>
    </div>
    <div class="summary-card name">
        <div class="summary-value">{counts.get("ORG", 0) + counts.get("NAME", 0)}</div>
        <div class="summary-label">ORG + NAME</div>
    </div>
    <div class="summary-card geo">
        <div class="summary-value">{counts.get("GEO", 0)}</div>
        <div class="summary-label">География</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        table_rows = []

        for result in batch_results:
            counts_row = Counter(
                e["label"]
                for e in result.get("entities", [])
            )

            table_rows.append({
                "ID": result["id"],
                "Текст": (
                    result["text"][:120] + "…"
                    if len(result["text"]) > 120
                    else result["text"]
                ),
                "Сущностей": len(result.get("entities", [])),
                "ORG": counts_row.get("ORG", 0),
                "NAME": counts_row.get("NAME", 0),
                "GEO": counts_row.get("GEO", 0),
                "Ошибка": result.get("error", ""),
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            hide_index=True,
            use_container_width=True,
        )

        batch_export_cols = st.columns(2)

        batch_export_cols[0].download_button(
            "Скачать результат CSV",
            data=batch_csv(batch_results),
            file_name="batch_entities.csv",
            mime="text/csv",
            use_container_width=True,
        )

        batch_export_cols[1].download_button(
            "Скачать результат JSONL",
            data=batch_jsonl(batch_results),
            file_name="batch_entities.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )


st.markdown(
    """
<div style="
    margin-top:2.8rem;
    text-align:center;
    color:#969DA7;
    font-size:.76rem;
">
    Распознавание ORG · NAME · GEO в узбекских текстах
</div>
""",
    unsafe_allow_html=True,
)
