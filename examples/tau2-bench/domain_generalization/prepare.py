"""Preserve source rows, split domains, and measure the training tokenizer/mask."""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

COUNTS = dict(airline=13783, retail=13614, telecom=2979, banking=5672)
SHARED = Path('/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments')
SOURCES = [SHARED / 'tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl',
           SHARED / 'tau2-banking-simplify-qwen38-v1/data/full/banking_simplified_sft.jsonl']


def read_sources(paths):
    rows = []
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    domain = row['metadata']['domain'].replace('banking_knowledge', 'banking')
                    if domain not in COUNTS:
                        raise ValueError(f'unknown domain {domain}')
                    messages = row['messages']
                    if messages[-1]['role'] != 'assistant':
                        raise ValueError('last message is not Assistant')
                    if [i for i, m in enumerate(messages) if m['role'] == 'assistant' and m.get('step_loss_mask', 1) == 1] != [len(messages)-1]:
                        raise ValueError('expected last-Assistant-only supervision')
                    if not (messages[-1].get('content') or messages[-1].get('tool_calls')):
                        raise ValueError('empty target')
                except Exception as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                rows.append(dict(domain=domain, source=str(path), line=line_number, raw=line.rstrip('\n')))
    return rows


def partition(rows, seed=1234):
    arms = {d: [r for r in rows if r['domain'] == d] for d in COUNTS}
    arms['mixed'] = list(rows)
    random.Random(seed).shuffle(arms['mixed'])
    return arms


def coverage(rows, epochs=2, batch=16, seed=1234):
    updates = epochs * (len(rows) // batch)
    samples = updates * batch
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    second = list(range(len(rows)))
    random.Random(seed + 1).shuffle(second)
    consumed = (order + second)[:samples]
    counts = Counter(consumed)
    return dict(updates=updates, consumed_samples=samples,
                coverage_counts=dict(Counter(counts.get(i, 0) for i in range(len(rows)))),
                input_tokens=sum(rows[i]['tokens'] for i in consumed) if 'tokens' in rows[0] else None,
                supervised_tokens=sum(rows[i]['supervised_tokens'] for i in consumed) if 'supervised_tokens' in rows[0] else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=2, default=SOURCES)
    parser.add_argument('--tokenizer', default='/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507')
    parser.add_argument('--tokenize', action='store_true', help='Run in the training container')
    args = parser.parse_args()
    rows = read_sources(args.sources)
    counts = Counter(r['domain'] for r in rows)
    if counts != COUNTS:
        raise ValueError(f'source counts changed: {counts}, expected {COUNTS}')
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = args.output.parent / 'benchmark' / 'tau2'
    if not benchmark.exists():
        shutil.copytree('/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/data/tau2', benchmark)
    if args.tokenize:
        from transformers import AutoTokenizer
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        generator = MultiTurnLossMaskGenerator(tokenizer, 'qwen3_full')
        for index, record in enumerate(rows):
            row = json.loads(record['raw'])
            try:
                ids, mask, spans = generator.gen_multi_turn_loss_mask_qwen3_full(
                    row['messages'], tools=row.get('tools'), return_assistant_spans=True)
                target = spans[-1]
                start, end = target['token_start'], target['token_end']
                if len(ids) > 16384:
                    raise ValueError(f'{len(ids)} tokens exceeds 16384')
                if not sum(mask) or any(mask[:start]) or not all(mask[start:end]):
                    raise ValueError('target span supervision mismatch')
                if not any(t == tokenizer.eos_token_id and m for t, m in zip(ids[start:end], mask[start:end])):
                    raise ValueError('target EOS not supervised')
                record.update(tokens=len(ids), supervised_tokens=sum(mask))
            except Exception as exc:
                raise ValueError(f"{record['source']}:{record['line']}: {exc}") from exc
            if index % 1000 == 0:
                print(f'tokenized {index}/{len(rows)}', flush=True)
        with (args.output / 'token_lengths.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps({k:v for k,v in row.items() if k != 'raw'}) + '\n')
    arms = partition(rows)
    if args.tokenize:
        arms['smoke'] = [r for d in COUNTS for r in sorted(arms[d], key=lambda r: -r['tokens'])[:8]]
    stats = {}
    for arm, subset in arms.items():
        with (args.output / f'{arm}.jsonl').open('w') as stream:
            for row in subset:
                stream.write(row['raw'] + '\n')
        stats[arm] = dict(rows=len(subset), **coverage(subset, epochs=1 if arm == 'smoke' else 2))
    (args.output / ('token_stats.json' if args.tokenize else 'data_stats.json')).write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
