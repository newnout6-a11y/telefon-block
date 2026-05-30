"""
Validates feature schema parity between Python and Android Kotlin code.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CALL_FEATURES = os.path.join(ROOT, 'app', 'src', 'main', 'java', 'com', 'antispam', 'blocker', 'domain', 'scoring', 'CallFeatures.kt')
DECISION_TRACKER = os.path.join(ROOT, 'app', 'src', 'main', 'java', 'com', 'antispam', 'blocker', 'domain', 'tracking', 'DecisionTracker.kt')


def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_feature_count(kotlin_text):
    match = re.search(r'const\s+val\s+FEATURE_COUNT\s*=\s*(\d+)', kotlin_text)
    return int(match.group(1)) if match else None


def extract_tracker_names(kotlin_text):
    marker = 'val FEATURE_NAMES: List<String> = listOf('
    start = kotlin_text.find(marker)
    if start < 0:
        return []
    start += len(marker)
    end = kotlin_text.find(')', start)
    if end < 0:
        return []
    block = kotlin_text[start:end]
    return re.findall(r'"([A-Za-z0-9_]+)"', block)


def main():
    call_text = read(CALL_FEATURES)
    tracker_text = read(DECISION_TRACKER)
    count = extract_feature_count(call_text)
    tracker_names = extract_tracker_names(tracker_text)

    errors = []
    if count != len(COMPACT_FEATURES):
        errors.append(f'CallFeatures.FEATURE_COUNT={count}, python={len(COMPACT_FEATURES)}')
    if tracker_names != COMPACT_FEATURES:
        errors.append('DecisionTracker.FEATURE_NAMES does not match ru_metadata_features.COMPACT_FEATURES')
        errors.append(f'python={COMPACT_FEATURES}')
        errors.append(f'kotlin={tracker_names}')

    if errors:
        print('SCHEMA VALIDATION FAILED')
        for err in errors:
            print(f'- {err}')
        raise SystemExit(1)

    print(f'SCHEMA OK: {len(COMPACT_FEATURES)} features')


if __name__ == '__main__':
    main()
