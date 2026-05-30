"""
Data quality validator for RF metadata pipeline.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES
from ru_number_normalizer import normalize_ru_phone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'datasets', 'ru', 'raw')
PROCESSED_DIR = os.path.join(ROOT, 'datasets', 'ru', 'processed')


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def validate_phone_csv(path, number_columns=('normalized_number', 'phone', 'number')):
    rows = read_csv(path)
    issues = []
    seen = Counter()
    for idx, row in enumerate(rows, start=2):
        raw = ''
        for col in number_columns:
            if row.get(col):
                raw = row[col]
                break
        digits = ''.join(ch for ch in raw if ch.isdigit())
        normalized = digits if digits in {'101', '102', '103', '104', '112'} else normalize_ru_phone(raw, reject_non_ru=False)
        if not normalized:
            issues.append((idx, 'invalid_phone', raw))
        else:
            seen[normalized] += 1
    duplicates = {n: c for n, c in seen.items() if c > 1}
    return {
        'path': path,
        'rows': len(rows),
        'invalid': issues[:30],
        'invalid_count': len(issues),
        'duplicate_count': len(duplicates),
        'duplicates_preview': dict(list(duplicates.items())[:20]),
    }


def validate_tflite_features(path):
    rows = read_csv(path)
    issues = []
    expected = COMPACT_FEATURES + ['label']
    if not rows:
        return {'path': path, 'rows': 0, 'issues': ['empty_or_missing']}
    header = list(rows[0].keys())
    if header != expected:
        issues.append(f'header_mismatch expected={expected} got={header}')
    labels = Counter()
    for idx, row in enumerate(rows, start=2):
        try:
            labels[int(float(row.get('label', -1)))] += 1
        except Exception:
            issues.append(f'bad_label line={idx} value={row.get("label")}')
        for feature in COMPACT_FEATURES:
            try:
                value = float(row.get(feature, 'nan'))
                if not 0.0 <= value <= 1.0:
                    issues.append(f'feature_out_of_range line={idx} feature={feature} value={value}')
            except Exception:
                issues.append(f'bad_feature line={idx} feature={feature} value={row.get(feature)}')
            if len(issues) > 50:
                break
    return {'path': path, 'rows': len(rows), 'labels': dict(labels), 'issues': issues[:50]}


def print_report(report):
    failed = False
    for item in report:
        print(f'\n== {item["path"]}')
        for key, value in item.items():
            if key == 'path':
                continue
            print(f'{key}: {value}')
        if item.get('invalid_count', 0) > 0 or item.get('issues'):
            failed = True
    return failed


def main():
    parser = argparse.ArgumentParser(description='Validate RF data files')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    files = [
        os.path.join(RAW_DIR, 'whitelist_official_ru.csv'),
        os.path.join(RAW_DIR, 'ru_reputation_raw.csv'),
        os.path.join(RAW_DIR, 'reviews_neberitrubku.csv'),
        os.path.join(RAW_DIR, 'reviews_zvonili.csv'),
        os.path.join(RAW_DIR, 'blacklist_moshelovka.csv'),
        os.path.join(PROCESSED_DIR, 'ru_tflite_features.csv'),
    ]
    report = []
    for path in files:
        if path.endswith('ru_tflite_features.csv'):
            report.append(validate_tflite_features(path))
        elif os.path.exists(path):
            report.append(validate_phone_csv(path))
        else:
            report.append({'path': path, 'rows': 0, 'missing': True})

    failed = print_report(report)
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
