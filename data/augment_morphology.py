"""Generate controlled Uzbek case-suffix examples for clean entity surfaces."""

from __future__ import annotations

import argparse
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from augmentation_common import JsonObject, read_jsonl, synthetic_record, unique_records, write_jsonl, write_report


LATIN_TEMPLATES = {
    "GEO": [
        ("da", "{entity} yangi loyiha boshlandi."),
        ("dan", "{entity} delegatsiya yetib keldi."),
        ("DATIVE", "{entity} safar rejalashtirildi."),
        ("dagi", "{entity} tadbir muvaffaqiyatli yakunlandi."),
        ("ning", "{entity} markazida yangi bino ochildi."),
    ],
    "NAME": [
        ("ning", "{entity} tarjimai holi oʻrganildi."),
        ("DATIVE", "{entity} bagʻishlangan maqola chop etildi."),
        ("ni", "Tadqiqotchilar {entity} maqolada tilga oldi."),
        ("dan", "Maqolada {entity} iqtibos keltirildi."),
    ],
    "ORG": [
        ("ning", "{entity} matbuot xizmati xabar berdi."),
        ("DATIVE", "{entity} rasmiy soʻrov yuborildi."),
        ("dan", "{entity} javob olindi."),
        ("da", "{entity} yangi loyiha taqdim etildi."),
        ("ni", "{entity} hamkorlikka taklif qilishdi."),
    ],
}

CYRILLIC_TEMPLATES = {
    "GEO": [
        ("да", "{entity} янги лойиҳа бошланди."),
        ("дан", "{entity} делегация етиб келди."),
        ("DATIVE", "{entity} сафар режалаштирилди."),
        ("даги", "{entity} тадбир муваффақиятли якунланди."),
        ("нинг", "{entity} марказида янги бино очилди."),
    ],
    "NAME": [
        ("нинг", "{entity} таржимаи ҳоли ўрганилди."),
        ("DATIVE", "{entity} бағишланган мақола чоп этилди."),
        ("ни", "Тадқиқотчилар {entity} мақолада тилга олди."),
        ("дан", "Мақолада {entity} иқтибос келтирилди."),
    ],
    "ORG": [
        ("нинг", "{entity} матбуот хизмати хабар берди."),
        ("DATIVE", "{entity} расмий сўров юборилди."),
        ("дан", "{entity} жавоб олинди."),
        ("да", "{entity} янги лойиҳа тақдим этилди."),
        ("ни", "{entity} ҳамкорликка таклиф қилишди."),
    ],
}

ALL_SUFFIXES = sorted(
    {
        suffix
        for templates in (LATIN_TEMPLATES, CYRILLIC_TEMPLATES)
        for values in templates.values()
        for suffix, _ in values
        if suffix != "DATIVE"
    }
    | {"ga", "ka", "qa", "га", "ка", "қа"},
    key=len,
    reverse=True,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate controlled Uzbek morphology examples.")
    parser.add_argument("--input", type=Path, default=root / "train.jsonl")
    parser.add_argument("--output", type=Path, default=root / "augmentation" / "morphology.jsonl")
    parser.add_argument("--min-frequency", type=int, default=3)
    parser.add_argument("--min-purity", type=float, default=1.0)
    parser.add_argument("--max-entities-per-label", type=int, default=0, help="0 means no limit")
    return parser.parse_args()


def script(surface: str) -> str | None:
    latin = cyrillic = False
    for char in surface:
        name = unicodedata.name(char, "")
        if char.isalpha() and "LATIN" in name:
            latin = True
        elif char.isalpha() and "CYRILLIC" in name:
            cyrillic = True
    if latin and not cyrillic:
        return "latin"
    if cyrillic and not latin:
        return "cyrillic"
    return None


def normalized(surface: str) -> str:
    return " ".join(surface.casefold().split())


def clean_surface(surface: str) -> bool:
    if not 2 <= len(surface) <= 80 or "\n" in surface or "\r" in surface:
        return False
    if (
        surface != surface.strip()
        or not surface[0].isalnum()
        or not surface[-1].isalnum()
        or "@" in surface
        or "#" in surface
        or "://" in surface
    ):
        return False
    return script(surface) in {"latin", "cyrillic"}


def visibly_inflected(surface: str, surface_script: str) -> bool:
    """Conservatively reject surfaces that already carry a case suffix.

    This intentionally sacrifices bases such as Canada/Honda: for synthetic
    data, losing a few valid bases is preferable to creating double suffixes.
    Administrative nominatives such as *tumani* are allowed.
    """

    value = normalized(surface).rstrip("'’‘ʼʻ")
    if surface_script == "latin":
        blocked = ("ning", "dagi", "dan", "ga", "ka", "qa", "da")
        accusative = "ni"
        nominative_exceptions = ("tumani",)
    else:
        blocked = ("нинг", "даги", "дан", "га", "ка", "қа", "да")
        accusative = "ни"
        nominative_exceptions = ("тумани",)
    if value.endswith(blocked):
        return True
    return value.endswith(accusative) and not value.endswith(nominative_exceptions)


def collect_surfaces(records: list[JsonObject], min_frequency: int, min_purity: float) -> dict[str, list[tuple[str, int]]]:
    by_surface: dict[str, Counter[str]] = defaultdict(Counter)
    display: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        for entity in record["entities"]:
            surface = record["text"][entity["start"] : entity["end"]]
            key = normalized(surface)
            by_surface[key][entity["label"]] += 1
            display[(key, entity["label"])][surface] += 1

    # This deliberately includes rare and ambiguous surfaces. It is used only
    # to recognize that a more frequent candidate is already an inflected form.
    all_attested_by_label: dict[str, set[str]] = defaultdict(set)
    for key, labels in by_surface.items():
        for label in labels:
            all_attested_by_label[label].add(key)

    selected: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, labels in by_surface.items():
        total = sum(labels.values())
        label, frequency = labels.most_common(1)[0]
        if frequency < min_frequency or frequency / total < min_purity:
            continue
        surface = display[(key, label)].most_common(1)[0][0]
        if clean_surface(surface):
            selected[label].append((surface, frequency))

    for label in list(selected):
        filtered = []
        for surface, frequency in selected[label]:
            value = normalized(surface)
            surface_script = script(surface)
            english_possessive = (
                len(value) > 3
                and value[-2] in "'’‘ʼʻ"
                and value[-1] == "s"
                and value[:-2] in all_attested_by_label[label]
            )
            already_inflected = any(
                value.endswith(suffix)
                and len(value) > len(suffix) + 1
                and value[: -len(suffix)].rstrip("'’‘ʼʻ") in all_attested_by_label[label]
                for suffix in ALL_SUFFIXES
            )
            if (
                not already_inflected
                and not english_possessive
                and surface_script is not None
                and not visibly_inflected(surface, surface_script)
            ):
                filtered.append((surface, frequency))
        selected[label] = sorted(filtered, key=lambda item: (-item[1], item[0]))
    return selected


def resolve_suffix(surface: str, suffix: str, surface_script: str) -> str:
    """Apply the standard Uzbek -ga/-ka/-qa dative allomorph."""

    if suffix != "DATIVE":
        return suffix
    last = surface.rstrip().casefold()[-1]
    if surface_script == "latin":
        return "ka" if last == "k" else "qa" if last == "q" else "ga"
    return "ка" if last == "к" else "қа" if last == "қ" else "га"


def generate(selected: dict[str, list[tuple[str, int]]], max_entities: int) -> Iterator[JsonObject]:
    for label, surfaces in selected.items():
        if max_entities:
            surfaces = surfaces[:max_entities]
        for surface, frequency in surfaces:
            surface_script = script(surface)
            templates = LATIN_TEMPLATES if surface_script == "latin" else CYRILLIC_TEMPLATES
            for suffix_code, template in templates[label]:
                suffix = resolve_suffix(surface, suffix_code, surface_script)
                inflected = surface + suffix
                text = template.format(entity=inflected)
                start = text.index(inflected)
                details = {
                    "label": label,
                    "base_surface": surface,
                    "suffix": suffix,
                    "script": surface_script,
                    "source_frequency": frequency,
                    "template": template,
                }
                yield synthetic_record(
                    text,
                    [{"label": label, "start": start, "end": start + len(inflected)}],
                    kind="morphology",
                    details=details,
                )


def main() -> int:
    args = parse_args()
    if args.min_frequency < 1 or not 0 < args.min_purity <= 1 or args.max_entities_per_label < 0:
        raise ValueError("invalid frequency, purity or entity limit")
    records = read_jsonl(args.input)
    selected = collect_surfaces(records, args.min_frequency, args.min_purity)
    count = write_jsonl(
        args.output,
        unique_records(generate(selected, args.max_entities_per_label)),
    )
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "generated_records": count,
        "candidate_bases": {label: len(values) for label, values in selected.items()},
        "min_frequency": args.min_frequency,
        "min_purity": args.min_purity,
        "note": "Synthetic pool; subsample during training and validate impact on real dev data.",
    }
    write_report(args.output.with_suffix(".report.json"), report)
    print(f"Generated {count} morphology records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
