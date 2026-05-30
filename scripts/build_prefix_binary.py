#!/usr/bin/env python3
"""PR-3: конвертер prefix_histogram*.json → компактный бинарный формат .phbin.

Зачем
-----
Сейчас `app/src/main/assets/prefix_histogram_7.json` весит ~22 МБ и
парсится org.json'ом на старте сервиса (`PrefixHistogramTable.load`).
JSON-парсер на Android делает много мелких аллокаций и держит всё в
памяти как `HashMap<String, JSONObject>` плюс boxed Float'ы. Это:
  * 22 МБ APK / OTA-payload,
  * ~150 МБ heap при парсинге,
  * 200-400 мс на cold start среднего телефона.

Бинарный формат `.phbin` решает все три проблемы:
  * детерминированная sorted-by-prefix лента,
  * 8 байт на запись (uint32 prefix + 3*uint8 quantized + uint8 reserved
    + uint16 seen_count),
  * читается на Android через MappedByteBuffer + binary search,
  * no JSON parsing, no per-entry boxing.

Формат `.phbin` (little-endian)
-------------------------------

Header (64 байта):
    offset  size  field
    ------  ----  -----
       0     4    magic "PHBN"  (0x50 0x48 0x42 0x4e)
       4     2    format_version (uint16, =1)
       6     2    prefix_length (uint16) — количество ASCII-символов в
                  оригинальном префиксе (например, 9 для +73012150).
                  Само хранение — в uint32, представляющем цифры после "+7"
                  как десятичное число с фиксированным числом digits =
                  prefix_length - 2 (так "+73012150" → digits=7, prefix_int=3012150).
       8     4    record_count (uint32)
      12     4    seen_log_norm (float32) — для совместимости с PrefixHistogramTable.lookup
      16     4    sample_size_saturation (float32)
      20     4    overall_block_rate (float32)
      24     4    overall_warn_rate (float32)
      28     4    overall_seen_count (uint32, для аудита; 0 если неизвестно)
      32    16    version_string (ASCII, 0-padded; truncate if longer)
      48    16    reserved (zeros)

Records (record_count × 12 bytes), отсортированы по prefix_int возрастанию:
    offset  size  field
    ------  ----  -----
       0     4    prefix_int (uint32 LE) — цифры префикса как число
       4     1    block_share_q (uint8) — round(block_share * 255)
       5     1    warn_share_q  (uint8)
       6     1    entropy_q     (uint8)
       7     1    reserved (=0)
       8     4    seen_count    (uint32 LE)

Total: 64 + 12*N bytes. Для prefix_histogram_7 (~700k записей) ~8.4 МБ
вместо 22 МБ JSON, **в 2.6× меньше**. Точность quantize 1/255 ≈ 0.4% —
на BLOCK precision не влияет (пороги работают на уровне 0.01).

Использование
-------------
    python3 scripts/build_prefix_binary.py \
        --input  app/src/main/assets/prefix_histogram_7.json \
        --output app/src/main/assets/prefix_histogram_7.phbin

    # Или сразу 3 файла одним вызовом:
    python3 scripts/build_prefix_binary.py --build-all

PR-3 пишет .phbin рядом с .json, не удаляя последний — это даёт
backward-compat: старые версии Android-кода продолжат читать JSON.
Когда новый PrefixHistogramTable.kt из этого PR попадает на устройство,
он пробует .phbin первым, .json — fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Tuple

PHBIN_MAGIC = b'PHBN'
PHBIN_VERSION = 1
HEADER_SIZE = 64
RECORD_SIZE = 12
VERSION_STRING_BYTES = 16

DEFAULT_TARGETS = [
    ('app/src/main/assets/prefix_histogram.json',   'app/src/main/assets/prefix_histogram.phbin'),
    ('app/src/main/assets/prefix_histogram_3.json', 'app/src/main/assets/prefix_histogram_3.phbin'),
    ('app/src/main/assets/prefix_histogram_7.json', 'app/src/main/assets/prefix_histogram_7.phbin'),
]


def quantize_unit(x: float) -> int:
    """Float [0, 1] → uint8. Round-to-nearest-even, clipped."""
    if x is None:
        return 0
    v = max(0.0, min(1.0, float(x)))
    return int(round(v * 255.0))


def prefix_to_int(prefix: str, expected_length: int) -> int:
    """+73012150 → 3012150 (uint32). Принимает префиксы с/без +7.

    Префикс должен начинаться с '+7' и состоять из ASCII-цифр после.
    Длина проверяется (must == expected_length).
    """
    if len(prefix) != expected_length:
        raise ValueError(f'prefix length mismatch: {prefix!r} ({len(prefix)}) vs {expected_length}')
    if not prefix.startswith('+7'):
        raise ValueError(f'prefix must start with "+7": {prefix!r}')
    digits = prefix[2:]
    if not digits.isdigit():
        raise ValueError(f'prefix has non-digits after "+7": {prefix!r}')
    n = int(digits)
    if n > 0xFFFFFFFF:
        raise ValueError(f'prefix too large for uint32: {prefix!r}')
    return n


def int_to_prefix(prefix_int: int, expected_length: int) -> str:
    """Обратная операция, для тестов."""
    digits_count = expected_length - 2  # без "+7"
    return '+7' + str(prefix_int).zfill(digits_count)


def build_records(prefixes: Dict[str, dict], expected_length: int) -> List[Tuple[int, int, int, int, int]]:
    """Собирает список записей (sorted by prefix_int).

    Returns list of tuples: (prefix_int, b_q, w_q, e_q, seen_count).
    """
    rows: List[Tuple[int, int, int, int, int]] = []
    skipped_bad_prefix = 0
    skipped_bad_seen = 0
    for prefix, entry in prefixes.items():
        try:
            pi = prefix_to_int(prefix, expected_length)
        except ValueError:
            skipped_bad_prefix += 1
            continue
        b = quantize_unit(entry.get('block_share', 0.0))
        w = quantize_unit(entry.get('warn_share', 0.0))
        e = quantize_unit(entry.get('entropy', 0.0))
        seen = int(entry.get('seen_count', 0))
        if seen < 0:
            skipped_bad_seen += 1
            continue
        if seen > 0xFFFFFFFF:
            seen = 0xFFFFFFFF  # cap, очень редкий случай
        rows.append((pi, b, w, e, seen))
    rows.sort(key=lambda r: r[0])
    if skipped_bad_prefix:
        print(f'  WARNING: skipped {skipped_bad_prefix} entries with malformed prefix')
    if skipped_bad_seen:
        print(f'  WARNING: skipped {skipped_bad_seen} entries with negative seen_count')
    return rows


def encode_phbin(meta: dict, records: List[Tuple[int, int, int, int, int]]) -> bytes:
    """Строит весь .phbin как bytes."""
    version_str = str(meta.get('version', '') or '').encode('ascii', errors='ignore')[:VERSION_STRING_BYTES]
    version_str = version_str.ljust(VERSION_STRING_BYTES, b'\x00')
    header = struct.pack(
        '<4sHHIffffI16s16s',
        PHBIN_MAGIC,
        PHBIN_VERSION,
        int(meta['prefix_length']),
        len(records),
        float(meta.get('seen_log_norm', 1.0) or 1.0),
        float(meta.get('sample_size_saturation', 30.0) or 30.0),
        float(meta.get('overall_block_rate', 0.0) or 0.0),
        float(meta.get('overall_warn_rate', 0.0) or 0.0),
        int(meta.get('overall_seen_count', 0) or 0),
        version_str,
        b'\x00' * VERSION_STRING_BYTES,  # reserved
    )
    assert len(header) == HEADER_SIZE, f'header size {len(header)} != {HEADER_SIZE}'
    body = bytearray()
    for pi, bq, wq, eq, seen in records:
        body += struct.pack('<IBBBBI', pi, bq, wq, eq, 0, seen)
    return bytes(header) + bytes(body)


def convert(input_path: str, output_path: str) -> dict:
    """Конвертирует один JSON в .phbin. Возвращает stats."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    prefixes = data.get('prefixes', {})
    if not isinstance(prefixes, dict):
        raise SystemExit(f'{input_path}: "prefixes" is not an object')
    expected_length = int(data.get('prefix_length', 6))
    records = build_records(prefixes, expected_length)
    blob = encode_phbin(data, records)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(blob)
    in_size = os.path.getsize(input_path)
    out_size = len(blob)
    ratio = in_size / max(out_size, 1)
    print(
        f'  {input_path}: {len(prefixes)} prefixes, prefix_length={expected_length}\n'
        f'    JSON  {in_size:>12,} bytes ({in_size/1024/1024:.2f} MB)\n'
        f'    PHBIN {out_size:>12,} bytes ({out_size/1024/1024:.2f} MB)  → {ratio:.1f}× smaller'
    )
    return {
        'input': input_path,
        'output': output_path,
        'in_bytes': in_size,
        'out_bytes': out_size,
        'compression_ratio': ratio,
        'records': len(records),
        'prefix_length': expected_length,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Convert prefix_histogram*.json to compact .phbin')
    ap.add_argument('--input', help='Input .json file')
    ap.add_argument('--output', help='Output .phbin file')
    ap.add_argument('--build-all', action='store_true',
                    help='Convert all 3 default prefix histogram assets in-place. '
                         'Looks for prefix_histogram*.json in app/src/main/assets/.')
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    targets: List[Tuple[str, str]] = []
    if args.build_all:
        for src, dst in DEFAULT_TARGETS:
            src_abs = os.path.join(repo_root, src)
            dst_abs = os.path.join(repo_root, dst)
            if os.path.exists(src_abs):
                targets.append((src_abs, dst_abs))
            else:
                print(f'  skip (missing): {src}')
    elif args.input and args.output:
        targets.append((args.input, args.output))
    else:
        ap.error('Either --input/--output OR --build-all is required')

    total_in = 0
    total_out = 0
    for src, dst in targets:
        st = convert(src, dst)
        total_in += st['in_bytes']
        total_out += st['out_bytes']

    if total_in:
        print(
            f'\nTotal: {total_in:,} → {total_out:,} bytes  '
            f'({(total_in - total_out)/1024/1024:.2f} MB saved, '
            f'{total_in/total_out:.1f}× compression)'
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
