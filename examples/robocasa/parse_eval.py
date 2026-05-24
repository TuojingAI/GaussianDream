#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_summary(path: Path):
    if path.is_dir():
        candidates = sorted(path.glob('**/summary.json'))
        if not candidates:
            raise FileNotFoundError(f'No summary.json found under {path}')
        path = candidates[-1]
    if path.name != 'summary.json':
        raise ValueError('Please pass a summary.json file or a directory containing one')
    data = json.loads(path.read_text(encoding='utf-8'))
    return path, data


def main():
    parser = argparse.ArgumentParser(description='Print RoboCasa H50 per-family success rates from summary.json')
    parser.add_argument('path', help='Path to summary.json or a directory containing evaluation runs')
    args = parser.parse_args()

    summary_path, data = load_summary(Path(args.path))
    print(f'summary_path={summary_path}')
    print(f'run_dir={data.get("run_dir", "")}')
    print(f'host={data.get("host", "")} port={data.get("port", "")}')
    print(f'micro_success_rate={data.get("micro_success_rate", 0.0):.3f}')
    print(f'macro_success_rate={data.get("macro_success_rate", 0.0):.3f}')
    print(f'successes={data.get("successes", 0)}/{data.get("num_episodes", 0)}')
    print('--- per_family ---')
    per_family = data.get('per_family', {})
    ranked = sorted(per_family.items(), key=lambda kv: (-kv[1].get('success_rate', 0.0), kv[0]))
    for family, row in ranked:
        print(
            f'{family}\t{row.get("successes", 0)}/{row.get("episodes", 0)}\t{row.get("success_rate", 0.0):.3f}'
        )


if __name__ == '__main__':
    main()
