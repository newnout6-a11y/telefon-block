"""
Data quality checker for SpamBlocker raw/processed CSV files.

Validates: whitelist, blacklist, reviews, processed features, label distribution,
missing values, duplicate numbers, schema consistency.
"""

import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES
from ru_number_normalizer import normalize_ru_phone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'datasets', 'ru', 'raw')
PROC_DIR = os.path.join(ROOT, 'datasets', 'ru', 'processed')

LABELS = {'ALLOW', 'WARN', 'BLOCK'}


def load_csv_rows(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def check_raw_whitelist() -> List[str]:
    issues = []
    rows = load_csv_rows(os.path.join(RAW_DIR, 'whitelist_official_ru.csv'))
    if not rows:
        issues.append('whitelist_official_ru.csv: empty or missing')
        return issues
    seen = set()
    for i, row in enumerate(rows):
        num = row.get('normalized_number', '').strip()
        if not num:
            issues.append(f'whitelist row {i}: empty normalized_number')
            continue
        norm = normalize_ru_phone(num, reject_non_ru=True)
        if not norm:
            issues.append(f'whitelist row {i}: invalid RU number "{num}"')
        if norm in seen:
            issues.append(f'whitelist row {i}: duplicate number {norm}')
        seen.add(norm)
    if len(rows) < 10:
        issues.append(f'whitelist: only {len(rows)} entries (expected 50+)')
    return issues


def check_raw_blacklist() -> List[str]:
    issues = []
    for filename in ['blacklist_moshelovka.csv', 'blacklist_spravportal.csv']:
        rows = load_csv_rows(os.path.join(RAW_DIR, filename))
        if not rows:
            issues.append(f'{filename}: empty or missing')
            continue
        seen = set()
        for i, row in enumerate(rows):
            num = row.get('normalized_number', '').strip()
            if not num:
                issues.append(f'{filename} row {i}: empty normalized_number')
                continue
            norm = normalize_ru_phone(num, reject_non_ru=True)
            if not norm:
                issues.append(f'{filename} row {i}: invalid RU number "{num}"')
            if norm in seen:
                issues.append(f'{filename} row {i}: duplicate {norm}')
            seen.add(norm)
    return issues


def check_raw_reviews() -> List[str]:
    issues = []
    for filename in ['reviews_neberitrubku.csv', 'reviews_zvonili.csv']:
        rows = load_csv_rows(os.path.join(RAW_DIR, filename))
        if not rows:
            issues.append(f'{filename}: empty or missing')
            continue
        for i, row in enumerate(rows):
            num = row.get('normalized_number', '').strip()
            if not num:
                issues.append(f'{filename} row {i}: empty normalized_number')
                break
            neg = row.get('negative_count', '0')
            pos = row.get('positive_count', '0')
            try:
                int(neg)
                int(pos)
            except ValueError:
                issues.append(f'{filename} row {i}: non-integer count neg={neg} pos={pos}')
                break
    return issues


def check_processed_features() -> List[str]:
    issues = []
    path = os.path.join(PROC_DIR, 'ru_tflite_features.csv')
    if not os.path.exists(path):
        issues.append('ru_tflite_features.csv: missing')
        return issues

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        issues.append('ru_tflite_features.csv: no data rows')
        return issues

    missing_features = [f for f in COMPACT_FEATURES if f not in header]
    if missing_features:
        issues.append(f'ru_tflite_features.csv: missing columns: {missing_features}')

    if 'label' not in header:
        issues.append('ru_tflite_features.csv: missing "label" column')
    else:
        label_counts = Counter()
        for row in rows:
            label_val = row.get('label', '').strip()
            try:
                label_id = int(float(label_val))
                label_counts[label_id] += 1
            except ValueError:
                issues.append(f'ru_tflite_features.csv: invalid label "{label_val}"')
        if label_counts:
            total = sum(label_counts.values())
            for cls_id in range(3):
                count = label_counts.get(cls_id, 0)
                ratio = count / total if total > 0 else 0
                if count == 0:
                    issues.append(f'ru_tflite_features.csv: class {cls_id} has 0 samples')
                elif ratio < 0.05:
                    issues.append(f'ru_tflite_features.csv: class {cls_id} only {ratio:.1%} of data (severe imbalance)')

    nan_count = 0
    for row in rows:
        for feat in COMPACT_FEATURES:
            val = row.get(feat, '').strip()
            if val == '' or val.lower() in ('nan', 'none', 'null'):
                nan_count += 1
    if nan_count > 0:
        issues.append(f'ru_tflite_features.csv: {nan_count} missing/NaN values')

    return issues


def check_label_consistency() -> List[str]:
    issues = []
    labeled_path = os.path.join(PROC_DIR, 'ru_numbers_labeled.csv')
    features_path = os.path.join(PROC_DIR, 'ru_tflite_features.csv')
    if not os.path.exists(labeled_path) or not os.path.exists(features_path):
        return issues

    labeled_rows = load_csv_rows(labeled_path)
    label_set = set()
    for row in labeled_rows:
        lbl = row.get('label', '').strip().upper()
        if lbl not in LABELS:
            issues.append(f'ru_numbers_labeled.csv: invalid label "{lbl}"')
        label_set.add(lbl)

    if not label_set.intersection({'BLOCK', 'WARN'}):
        issues.append('ru_numbers_labeled.csv: no BLOCK or WARN labels — model cannot learn')

    return issues


def check_cross_source_conflicts() -> List[str]:
    issues = []
    whitelist_nums = set()
    for row in load_csv_rows(os.path.join(RAW_DIR, 'whitelist_official_ru.csv')):
        norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
        if norm:
            whitelist_nums.add(norm)

    for bl_file in ['blacklist_moshelovka.csv', 'blacklist_spravportal.csv']:
        for row in load_csv_rows(os.path.join(RAW_DIR, bl_file)):
            norm = normalize_ru_phone(row.get('normalized_number', ''), reject_non_ru=True)
            if norm and norm in whitelist_nums:
                issues.append(f'CONFLICT: {norm} in both whitelist and {bl_file}')

    return issues


def main():
    strict = '--strict' in sys.argv
    all_issues = []

    all_issues.extend(check_raw_whitelist())
    all_issues.extend(check_raw_blacklist())
    all_issues.extend(check_raw_reviews())
    all_issues.extend(check_processed_features())
    all_issues.extend(check_label_consistency())
    all_issues.extend(check_cross_source_conflicts())

    if not all_issues:
        print('DATA QUALITY OK — no issues found')
    else:
        print(f'DATA QUALITY: {len(all_issues)} issue(s) found')
        for issue in all_issues:
            print(f'  - {issue}')
        if strict:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
