"""Normalize internal apostrophe variants without changing string length.

External quote marks are preserved. English contractions and possessives are
preserved by default so that multilingual examples are not Uzbek-normalized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from augmentation_common import read_jsonl, validate_record, write_jsonl, write_report


CANONICAL = "ʻ"  # U+02BB MODIFIER LETTER TURNED COMMA; dominant in this dataset.
VARIANTS = {"'", "’", "‘", "ʼ", "`", "´", "ʹ", "ʾ", "ʿ", "＇"}
ENGLISH_ENDINGS = {"s", "t", "re", "ve", "ll", "d", "m"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Normalize internal apostrophes in JSONL NER data.")
    parser.add_argument("--inputs", type=Path, nargs="+", default=[root / "train.jsonl", root / "dev.jsonl"])
    parser.add_argument("--output-dir", type=Path, default=root / "preprocessing")
    parser.add_argument(
        "--normalize-english",
        action="store_true",
        help="Also normalize apostrophes that look like English contractions/possessives.",
    )
    return parser.parse_args()


def word_bounds(text: str, index: int) -> tuple[int, int]:
    left = index
    while left > 0 and (text[left - 1].isalnum() or text[left - 1] in VARIANTS):
        left -= 1
    right = index + 1
    while right < len(text) and (text[right].isalnum() or text[right] in VARIANTS):
        right += 1
    return left, right


def looks_english(text: str, index: int) -> bool:
    left, right = word_bounds(text, index)
    before, after = text[left:index], text[index + 1 : right]
    if after.casefold() in ENGLISH_ENDINGS and before.isascii() and after.isascii():
        return True
    # O'Connor, D'Angelo and similar names are not Uzbek o'/g' spellings.
    if before.casefold() in {"o", "d"} and after[:1].isupper() and after.isascii():
        return True
    return False


def normalize_text(text: str, *, preserve_english: bool) -> tuple[str, Counter[str]]:
    chars = list(text)
    changes: Counter[str] = Counter()
    for index, char in enumerate(chars):
        if char not in VARIANTS or char == CANONICAL:
            continue
        if index == 0 or index + 1 == len(chars):
            continue
        if not (chars[index - 1].isalnum() and chars[index + 1].isalnum()):
            continue  # external quote, not an internal apostrophe
        if preserve_english and looks_english(text, index):
            continue
        chars[index] = CANONICAL
        changes[char] += 1
    return "".join(chars), changes


def normalize_record(record: dict[str, Any], *, preserve_english: bool) -> tuple[dict[str, Any], Counter[str], int]:
    normalized, changes = normalize_text(record["text"], preserve_english=preserve_english)
    if len(normalized) != len(record["text"]):
        raise AssertionError("apostrophe normalization must preserve text length")
    entity_changes = sum(
        record["text"][entity["start"] : entity["end"]]
        != normalized[entity["start"] : entity["end"]]
        for entity in record["entities"]
    )
    result = {"hash": record["hash"], "text": normalized, "entities": record["entities"]}
    validate_record(result, record["hash"])
    return result, changes, entity_changes


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.inputs:
        records = read_jsonl(input_path)
        output_records = []
        changes: Counter[str] = Counter()
        changed_documents = 0
        changed_entities = 0
        for record in records:
            normalized, record_changes, record_entity_changes = normalize_record(
                record,
                preserve_english=not args.normalize_english,
            )
            output_records.append(normalized)
            changes.update(record_changes)
            changed_documents += normalized["text"] != record["text"]
            changed_entities += record_entity_changes

        output_path = args.output_dir / input_path.name
        count = write_jsonl(output_path, output_records)
        report = {
            "input": str(input_path),
            "output": str(output_path),
            "records": count,
            "changed_documents": changed_documents,
            "changed_entities": changed_entities,
            "replacement_counts": dict(changes),
            "canonical_apostrophe": CANONICAL,
            "length_preserving": True,
            "english_like_apostrophes_preserved": not args.normalize_english,
        }
        write_report(args.output_dir / f"{input_path.stem}_apostrophe_report.json", report)
        print(f"{input_path.name}: {changed_documents}/{count} documents changed -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
