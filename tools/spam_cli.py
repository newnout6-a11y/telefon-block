"""
One-command SpamBlocker developer CLI.

Examples:
  python tools/spam_cli.py doctor
  python tools/spam_cli.py build-dataset --smoke-synthetic 80
  python tools/spam_cli.py train --export-tflite --plots
  python tools/spam_cli.py train --optuna-trials 20
  python tools/spam_cli.py export
  python tools/spam_cli.py drift --reference datasets/ru/processed/ru_tflite_features.csv
  python tools/spam_cli.py quality
  python tools/spam_cli.py android-build
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r'C:\Users\Redmi\AppData\Local\Programs\Python\Python312\python.exe')
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

ASSETS = ROOT / 'app' / 'src' / 'main' / 'assets'
TFLITE_MODEL = ASSETS / 'spam_model.tflite'
MODEL_CARD = ASSETS / 'model_card.json'


def c(text, code):
    return f'\033[{code}m{text}\033[0m' if os.environ.get('NO_COLOR') is None else text


def ok(text): print(c(f'  ✔ {text}', '32'))
def warn(text): print(c(f'  ⚠ {text}', '33'))
def fail(text): print(c(f'  ✖ {text}', '31'))
def info(text): print(c(f'  ▶ {text}', '36'))
def header(text): print(c(f'\n  {text}', '1;36'))
def dim(text): print(c(f'    {text}', '90'))


def run(cmd, check=True, env=None):
    info(' '.join(map(str, cmd)))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(cmd, cwd=ROOT, env=merged_env, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def java_env():
    env = {
        'JAVA_HOME': r'C:\Program Files\Android\Android Studio\jbr',
        'TEMP': str(Path.home() / 'tmp-gradle'),
        'TMP': str(Path.home() / 'tmp-gradle'),
    }
    Path(env['TEMP']).mkdir(parents=True, exist_ok=True)
    return env


def doctor(args):
    header('SpamBlocker Doctor')
    checks = [
        ('python', PYTHON.exists(), str(PYTHON)),
        ('gradlew', (ROOT / 'gradlew.bat').exists(), ''),
        ('android assets', ASSETS.exists(), ''),
        ('ru raw dir', (ROOT / 'datasets' / 'ru' / 'raw').exists(), ''),
        ('ru processed dir', (ROOT / 'datasets' / 'ru' / 'processed').exists(), ''),
        ('ru reports dir', (ROOT / 'datasets' / 'ru' / 'reports').exists(), ''),
    ]
    for name, passed, detail in checks:
        if passed:
            ok(f'{name} {detail}')
        else:
            fail(name)

    header('Schema validation')
    run([str(PYTHON), 'scripts/validate_feature_schema.py'], check=True)

    header('Data validation')
    run([str(PYTHON), 'scripts/validate_ru_data.py'], check=False)

    header('Model status')
    if TFLITE_MODEL.exists():
        ok(f'TFLite model: {TFLITE_MODEL.stat().st_size:,} bytes')
    else:
        warn('TFLite model missing — app will use rule fallback')
    if MODEL_CARD.exists():
        try:
            card = json.loads(MODEL_CARD.read_text(encoding='utf-8'))
            ok(f'Model card: v{card.get("version", "?")} features={card.get("feature_count", "?")} rows={card.get("rows", "?")}')
            bp = card.get('block_precision', 0)
            br = card.get('block_recall', 0)
            dim(f'BLOCK P={bp:.2f} R={br:.2f} best_model={card.get("best_model", "?")}')
        except Exception:
            warn('model_card.json: parse error')
    else:
        warn('model_card.json missing')

    header('Python packages')
    package_imports = {
        'scikit-learn': 'sklearn',
        'numpy': 'numpy',
        'imbalanced-learn': 'imblearn',
        'optuna': 'optuna',
        'catboost': 'catboost',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
    }
    for pkg, import_name in package_imports.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, '__version__', '?')
            ok(f'{pkg} {ver}')
        except ImportError:
            warn(f'{pkg} not installed')


def build_dataset(args):
    cmd = [str(PYTHON), 'scripts/ru_metadata_dataset_builder.py']
    if args.smoke_synthetic:
        cmd += ['--smoke-synthetic', str(args.smoke_synthetic)]
    run(cmd)


def train(args):
    cmd = [str(PYTHON), 'scripts/train_ru_metadata_models.py']
    if args.export_tflite:
        cmd.append('--export-tflite')
    if args.allow_unsafe_export:
        cmd.append('--allow-unsafe-export')
    cmd += ['--min-block-precision', str(args.min_block_precision)]
    if args.no_smote:
        cmd.append('--no-smote')
    if args.optuna_trials > 0:
        cmd += ['--optuna-trials', str(args.optuna_trials)]
    if args.drift_reference:
        cmd += ['--drift-reference', args.drift_reference]
    if args.plots:
        cmd.append('--plots')
    run(cmd)


def kd_train(args):
    cmd = [str(PYTHON), 'scripts/train_kd_distillation.py']
    if args.data:
        cmd += ['--data', args.data]
    if args.teacher_train_per_class is not None:
        cmd += ['--teacher-train-per-class', str(args.teacher_train_per_class)]
    if args.student_train_per_class is not None:
        cmd += ['--student-train-per-class', str(args.student_train_per_class)]
    if args.optuna_trials is not None:
        cmd += ['--optuna-trials', str(args.optuna_trials)]
    if args.min_block_precision is not None:
        cmd += ['--min-block-precision', str(args.min_block_precision)]
    if args.min_cold_block_precision is not None:
        cmd += ['--min-cold-block-precision', str(args.min_cold_block_precision)]
    if args.val_frac is not None:
        cmd += ['--val-frac', str(args.val_frac)]
    if args.test_frac is not None:
        cmd += ['--test-frac', str(args.test_frac)]
    if args.pad_with_smote:
        cmd.append('--pad-with-smote')
    if args.no_use_full_train:
        cmd.append('--no-use-full-train')
    if args.allow_unsafe_export:
        cmd.append('--allow-unsafe-export')
    if args.seed is not None:
        cmd += ['--seed', str(args.seed)]
    run(cmd)


def export(args):
    header('Export TFLite model')
    cmd = [str(PYTHON), 'scripts/train_ru_metadata_models.py', '--export-tflite']
    if args.allow_unsafe_export:
        cmd.append('--allow-unsafe-export')
    cmd += ['--min-block-precision', str(args.min_block_precision)]
    if args.plots:
        cmd.append('--plots')
    run(cmd)
    if TFLITE_MODEL.exists():
        ok(f'Exported: {TFLITE_MODEL.stat().st_size:,} bytes → {TFLITE_MODEL}')
    else:
        fail('Export failed — model not found')


def drift(args):
    header('Drift detection')
    cmd = [str(PYTHON), 'scripts/train_ru_metadata_models.py', '--drift-reference', args.reference]
    if args.plots:
        cmd.append('--plots')
    run(cmd)


def quality(args):
    header('Data quality check')
    run([str(PYTHON), 'scripts/validate_ru_data.py', '--strict'])
    run([str(PYTHON), 'scripts/validate_feature_schema.py'])

    raw_dir = ROOT / 'datasets' / 'ru' / 'raw'
    proc_dir = ROOT / 'datasets' / 'ru' / 'processed'

    header('Raw data files')
    if raw_dir.exists():
        for f in sorted(raw_dir.glob('*.csv')):
            lines = sum(1 for _ in open(f, encoding='utf-8'))
            ok(f'{f.name}: {lines} lines')
    else:
        fail('Raw dir missing')

    header('Processed data files')
    if proc_dir.exists():
        for f in sorted(proc_dir.glob('*.csv')):
            lines = sum(1 for _ in open(f, encoding='utf-8'))
            ok(f'{f.name}: {lines} lines')
    else:
        fail('Processed dir missing')


def validate(args):
    run([str(PYTHON), 'scripts/validate_feature_schema.py'])
    run([str(PYTHON), 'scripts/validate_ru_data.py'] + (['--strict'] if args.strict else []))


def android_build(args):
    run([str(ROOT / 'gradlew.bat'), 'assembleDebug'], env=java_env())


def status(args):
    header('Git status')
    run(['git', 'status', '--short'], check=False)
    dim(f'Branch: ', end='')
    run(['git', 'branch', '--show-current'], check=False)

    header('Model')
    if TFLITE_MODEL.exists():
        ok(f'{TFLITE_MODEL.stat().st_size:,} bytes')
    else:
        warn('No TFLite model')
    if MODEL_CARD.exists():
        card = json.loads(MODEL_CARD.read_text(encoding='utf-8'))
        ok(f'v{card.get("version", "?")} best={card.get("best_model", "?")}')
    else:
        warn('No model card')


def collect(args):
    cmd = [
        str(PYTHON),
        'scripts/ru_collect_sources.py',
        '--source', args.source,
        '--candidates', args.candidates,
        '--limit', str(args.limit),
    ]
    run(cmd)


def predict(args):
    cmd = [str(PYTHON), 'scripts/spam_predict.py']
    if args.cold:
        cmd.append('--cold')
    if args.no_rules:
        cmd.append('--no-rules')
    if args.show_features:
        cmd.append('--show-features')
    if args.json:
        cmd.append('--json')
    cmd += list(args.numbers)
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description='SpamBlocker dev CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('doctor').set_defaults(func=doctor)

    p = sub.add_parser('build-dataset')
    p.add_argument('--smoke-synthetic', type=int, default=0)
    p.set_defaults(func=build_dataset)

    p = sub.add_parser('train')
    p.add_argument('--export-tflite', action='store_true')
    p.add_argument('--allow-unsafe-export', action='store_true')
    p.add_argument('--min-block-precision', type=float, default=0.90)
    p.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    p.add_argument('--optuna-trials', type=int, default=0, help='Optuna trials (0=off)')
    p.add_argument('--drift-reference', type=str, default=None, help='Production CSV for drift')
    p.add_argument('--plots', action='store_true', help='Generate plots')
    p.set_defaults(func=train)

    p = sub.add_parser('export')
    p.add_argument('--allow-unsafe-export', action='store_true')
    p.add_argument('--min-block-precision', type=float, default=0.90)
    p.add_argument('--plots', action='store_true')
    p.set_defaults(func=export)

    p = sub.add_parser('kd-train', help='Knowledge Distillation: CatBoost teacher → Keras MLP student → TFLite')
    p.add_argument('--data', type=str, default=None,
                   help='CSV/NPZ with compact features. NPZ must contain X and y arrays.')
    p.add_argument('--teacher-train-per-class', type=int, default=6000,
                   help='На скольких примерах каждого класса обучается teacher (legit + spam).')
    p.add_argument('--student-train-per-class', type=int, default=4000,
                   help='Подвыборка из teacher train для student.')
    p.add_argument('--optuna-trials', type=int, default=20,
                   help='Optuna trials в stage2 (lr/dropout/hidden). 0=пропустить.')
    p.add_argument('--min-block-precision', type=float, default=0.85)
    p.add_argument('--min-cold-block-precision', type=float, default=None)
    p.add_argument('--val-frac', type=float, default=None)
    p.add_argument('--test-frac', type=float, default=None)
    p.add_argument('--no-use-full-train', action='store_true',
                   help='Use stratified teacher/student samples instead of all train rows.')
    p.add_argument('--pad-with-smote', action='store_true',
                   help='Добивать teacher train SMOTE-ом до целевых цифр при нехватке данных.')
    p.add_argument('--allow-unsafe-export', action='store_true',
                   help='Экспортировать .tflite даже при провале sanity check.')
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=kd_train)

    p = sub.add_parser('drift')
    p.add_argument('--reference', type=str, required=True, help='Production CSV to compare against')
    p.add_argument('--plots', action='store_true')
    p.set_defaults(func=drift)

    sub.add_parser('quality').set_defaults(func=quality)

    p = sub.add_parser('validate')
    p.add_argument('--strict', action='store_true')
    p.set_defaults(func=validate)

    sub.add_parser('android-build').set_defaults(func=android_build)
    sub.add_parser('status').set_defaults(func=status)

    p = sub.add_parser('collect')
    p.add_argument('--source', choices=['neberitrubku', 'zvonili', 'moshelovka'], required=True)
    p.add_argument('--candidates', required=True)
    p.add_argument('--limit', type=int, default=100)
    p.set_defaults(func=collect)

    p = sub.add_parser('predict', help='Прогон одного или нескольких номеров через TFLite-модель.')
    p.add_argument('numbers', nargs='+', help='Номер(а) телефона.')
    p.add_argument('--cold', action='store_true',
                   help='Игнорировать lookup CSV; считать фичи как для неизвестного номера.')
    p.add_argument('--no-rules', action='store_true',
                   help='Отключить post-model rule engine.')
    p.add_argument('--show-features', action='store_true', help='Распечатать все 32 фичи.')
    p.add_argument('--json', action='store_true', help='JSON-вывод вместо текстового.')
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
