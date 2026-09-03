"""Generate unambiguous, controlled hard-negative multilingual examples."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterator

from augmentation_common import JsonObject, synthetic_record, unique_records, write_jsonl, write_report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate controlled hard-negative NER examples.")
    parser.add_argument("--output", type=Path, default=root / "augmentation" / "hard_negatives.jsonl")
    return parser.parse_args()


FAMILIES = {
    "uzbek_latin_pronouns": [
        ("Men {tail}", ["bu fikrga qoʻshilaman.", "javobni bilmayman.", "bugun uyda qolaman.", "bu haqda keyin gapiraman."]),
        ("Bu vazifani kim {tail}?", ["bajaradi", "tekshiradi", "tugatadi", "nazorat qiladi"]),
        ("Hech kim {tail}", ["savolga javob bermadi.", "xonaga kirmadi.", "bu fikrni tasdiqlamadi.", "kechikmadi."]),
    ],
    "uzbek_latin_homonyms": [
        ("{head}", [
            "Jamoa bu taklifga qarshi chiqdi.",
            "Ular raqibga qarshi yaxshi o‘ynadi.",
            "Bu real natija emas.",
            "Bizga real hisob-kitob kerak.",
            "Butun dunyo bu muammoni muhokama qilmoqda.",
            "Dunyo doim o‘zgarib turadi.",
        ]),
    ],
    "uzbek_latin_generic_roles": [
        ("{subject} {tail}", [
            "mijozlarga yangi xizmat taklif qildi.",
            "bugun rasmiy bayonot berdi.",
            "masalani ko‘rib chiqmoqda.",
            "yangi tartibni tushuntirdi.",
        ]),
    ],
    "uzbek_latin_generic_places": [
        ("{subject} {tail}", [
            "bugun juda gavjum edi.",
            "yangi yo‘l qurilmoqda.",
            "obodonlashtirish ishlari boshlandi.",
            "yog‘ingarchilik kutilmoqda.",
        ]),
    ],
    "uzbek_cyrillic": [
        ("{head}", [
            "Мен бу фикрга қўшиламан.",
            "Бу вазифани ким бажаради?",
            "Ҳеч ким саволга жавоб бермади.",
            "Жамоа бу таклифга қарши чиқди.",
            "Бу реал натижа эмас.",
            "Бутун дунё бу муаммони муҳокама қилмоқда.",
            "Маҳаллий банк янги хизмат таклиф қилди.",
            "Йирик компания расмий баёнот берди.",
            "Президент йиғилиш ўтказди.",
            "Доктор беморни кўрикдан ўтказди.",
            "Шаҳар маркази бугун гавжум эди.",
            "Туман ҳудудида йўл таъмирланмоқда.",
        ]),
    ],
    "russian": [
        ("{head}", [
            "Кто будет отвечать за эту задачу?",
            "Никто не подтвердил эту информацию.",
            "Мы выступаем против этого решения.",
            "Это реальный результат, а не прогноз.",
            "Весь мир обсуждает эту проблему.",
            "Местный банк предложил новую услугу.",
            "Крупная компания опубликовала отчёт.",
            "Президент провёл совещание.",
            "Доктор осмотрел пациента.",
            "Город сегодня был очень тихим.",
            "Район постепенно благоустраивают.",
        ]),
    ],
    "english": [
        ("{head}", [
            "Who will answer this question?",
            "Nobody confirmed the report.",
            "They voted against the proposal.",
            "This is a real result, not an estimate.",
            "The whole world is discussing the problem.",
            "A local bank announced new rates.",
            "A large company published its report.",
            "The president spoke at the meeting.",
            "The doctor examined the patient.",
            "The city was quiet this morning.",
            "The district is changing rapidly.",
        ]),
    ],
}

GENERIC_SUBJECTS = ["Mahalliy bank", "Yirik kompaniya", "Universitet", "Vazirlik", "Prezident", "Doktor", "Professor"]
GENERIC_PLACES = ["Shahar markazi", "Viloyat hududi", "Tuman markazi", "Koʻcha", "Mahalla"]


def expand_family(name: str, pattern: str, values: list[str]) -> Iterator[tuple[str, str]]:
    if "{tail}" in pattern and "{subject}" in pattern:
        subjects = GENERIC_PLACES if "places" in name else GENERIC_SUBJECTS
        for subject, tail in itertools.product(subjects, values):
            yield pattern.format(subject=subject, tail=tail), name
    elif "{tail}" in pattern:
        for tail in values:
            yield pattern.format(tail=tail), name
    elif "{subject}" in pattern:
        subjects = GENERIC_PLACES if "places" in name else GENERIC_SUBJECTS
        for subject, tail in itertools.product(subjects, values):
            yield pattern.format(subject=subject, tail=tail), name
    else:
        for text in values:
            yield pattern.format(head=text), name


def generate() -> Iterator[JsonObject]:
    for family, templates in FAMILIES.items():
        for pattern, values in templates:
            for text, generated_family in expand_family(family, pattern, values):
                yield synthetic_record(
                    text,
                    [],
                    kind="hard-negative",
                    details={
                        "family": generated_family,
                        "policy": "Controlled sentence containing only pronouns, common nouns, roles or contextually non-entity homonyms.",
                    },
                )


def main() -> int:
    args = parse_args()
    count = write_jsonl(args.output, unique_records(generate()))
    report = {
        "output": str(args.output),
        "generated_records": count,
        "families": sorted(FAMILIES),
        "warning": "Keep these as a small precision-oriented fraction of training; do not let templates dominate real data.",
    }
    write_report(args.output.with_suffix(".report.json"), report)
    print(f"Generated {count} controlled hard-negative records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
