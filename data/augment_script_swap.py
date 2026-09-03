"""Create conservative Cyrillic/Latin entity replacements.

Only replacements attested in the input data with the same entity label are
used. One entity occurrence is replaced per generated record.
"""

from __future__ import annotations

import argparse
import unicodedata
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from augmentation_common import (
    JsonObject,
    read_jsonl,
    replace_entities,
    unique_records,
    write_jsonl,
    write_report,
)


APOSTROPHES = str.maketrans({char: "ʻ" for char in "'’‘ʼ`´ʹʾʿ＇"})
CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "ʻ", "ы": "i", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "ў": "oʻ", "қ": "q", "ғ": "gʻ",
    "ҳ": "h", "ң": "ng", "ү": "u", "ө": "o",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate attested cross-script entity swaps.")
    parser.add_argument("--input", type=Path, default=root / "train.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "augmentation" / "script_swap",
        help="Directory for editor-friendly part-*.jsonl files.",
    )
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument(
        "--preview",
        type=Path,
        default=root / "augmentation" / "script_swap_preview.jsonl",
    )
    parser.add_argument("--preview-size", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional monolithic JSONL. Disabled by default because it is too large for many editors.",
    )
    parser.add_argument("--max-aliases", type=int, default=3)
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


def transliterate_cyrillic(surface: str) -> str:
    result: list[str] = []
    for char in surface.casefold():
        result.append(CYRILLIC_TO_LATIN.get(char, char))
    return "".join(result)


def canonical_key(surface: str) -> str:
    value = surface.translate(APOSTROPHES).casefold()
    if script(surface) == "cyrillic":
        value = transliterate_cyrillic(value)
    return " ".join(value.split())


def build_aliases(records: list[JsonObject]) -> dict[tuple[str, str, str], list[str]]:
    frequencies: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    scripts_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        for entity in record["entities"]:
            surface = record["text"][entity["start"] : entity["end"]]
            surface_script = script(surface)
            if surface_script is None:
                continue
            key = (canonical_key(surface), entity["label"])
            scripts_by_key[key].add(surface_script)
            frequencies[(key[0], key[1], surface_script)][surface] += 1

    aliases: dict[tuple[str, str, str], list[str]] = {}
    for (key, label), available_scripts in scripts_by_key.items():
        if available_scripts != {"latin", "cyrillic"}:
            continue
        for source_script, target_script in (("latin", "cyrillic"), ("cyrillic", "latin")):
            candidates = frequencies[(key, label, target_script)]
            aliases[(key, label, source_script)] = [
                surface for surface, _ in candidates.most_common()
            ]
    return aliases


def generate(records: list[JsonObject], aliases: dict[tuple[str, str, str], list[str]], max_aliases: int) -> Iterator[JsonObject]:
    for record in records:
        for entity_index, entity in enumerate(record["entities"]):
            surface = record["text"][entity["start"] : entity["end"]]
            source_script = script(surface)
            if source_script is None:
                continue
            key = (canonical_key(surface), entity["label"], source_script)
            for alias in aliases.get(key, [])[:max_aliases]:
                if alias == surface:
                    continue
                details = {
                    "entity_index": entity_index,
                    "label": entity["label"],
                    "source_surface": surface,
                    "replacement_surface": alias,
                    "source_script": source_script,
                    "target_script": script(alias),
                    "selection": "same transliterated key and label; both surfaces attested in train",
                }
                yield replace_entities(
                    record,
                    {entity_index: alias},
                    kind="script-swap",
                    details=details,
                )


def write_shards(output_dir: Path, records: Iterable[JsonObject], shard_size: int) -> tuple[int, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk: list[JsonObject] = []
    paths: list[Path] = []
    total = 0

    for record in records:
        chunk.append(record)
        if len(chunk) < shard_size:
            continue
        path = output_dir / f"part-{len(paths) + 1:05d}.jsonl"
        total += write_jsonl(path, chunk)
        paths.append(path)
        chunk = []

    if chunk:
        path = output_dir / f"part-{len(paths) + 1:05d}.jsonl"
        total += write_jsonl(path, chunk)
        paths.append(path)
    return total, paths


def main() -> int:
    args = parse_args()
    if args.max_aliases < 1:
        raise ValueError("--max-aliases must be positive")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    if args.preview_size < 0:
        raise ValueError("--preview-size cannot be negative")
    records = read_jsonl(args.input)
    aliases = build_aliases(records)
    count, shard_paths = write_shards(
        args.output_dir,
        unique_records(generate(records, aliases, args.max_aliases)),
        args.shard_size,
    )
    preview_count = 0
    if args.preview_size:
        preview_records = unique_records(generate(records, aliases, args.max_aliases))
        preview_count = write_jsonl(
            args.preview,
            islice(preview_records, args.preview_size),
        )

    monolithic_count = None
    if args.output is not None:
        monolithic_count = write_jsonl(
            args.output,
            unique_records(generate(records, aliases, args.max_aliases)),
        )
    report = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "shard_size": args.shard_size,
        "shards": [str(path) for path in shard_paths],
        "preview": str(args.preview) if args.preview_size else None,
        "preview_records": preview_count,
        "monolithic_output": str(args.output) if args.output is not None else None,
        "monolithic_records": monolithic_count,
        "generated_records": count,
        "alias_directions": len(aliases),
        "policy": "Only cross-script surfaces attested with the same label are substituted.",
    }
    report_path = args.output_dir / "report.json"
    write_report(report_path, report)
    print(
        f"Generated {count} conservative script-swap records in {len(shard_paths)} shards "
        f"-> {args.output_dir}"
    )
    if args.preview_size:
        print(f"Preview: {preview_count} records -> {args.preview}")
    if args.output is not None:
        print(f"Monolithic output: {monolithic_count} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
