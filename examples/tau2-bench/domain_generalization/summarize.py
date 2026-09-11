"""Paired four-domain, two-seed evaluation tables and task bootstrap CIs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

QUOTAS = dict(airline=20, retail=40, telecom=40, banking_knowledge=97)
ARMS = ('raw', 'mixed', 'airline', 'retail', 'telecom', 'banking')
METRICS = ('pass@1', 'pass@4(any)', 'pass^4')


def task_scores(simulations, expected_ids, trials=4):
    grouped = defaultdict(dict)
    for sim in simulations:
        task, trial = sim['task_id'], sim['trial']
        if trial in grouped[task]:
            raise ValueError(f'duplicate trial: {task}/{trial}')
        if sim.get('termination_reason') == 'infrastructure_error':
            raise ValueError(f'infrastructure failure: {task}/{trial}')
        reward = (sim.get('reward_info') or {}).get('reward')
        if reward not in (0, 1):
            raise ValueError(f'missing/nonbinary reward: {task}/{trial}')
        grouped[task][trial] = reward
    if set(grouped) != set(expected_ids):
        raise ValueError('missing or unexpected task IDs')
    scores = {}
    for task, results in sorted(grouped.items()):
        if set(results) != set(range(trials)):
            raise ValueError(f'missing or unexpected trials: {task}')
        values = list(results.values())
        scores[task] = [sum(values)/trials, float(any(values)), float(all(values))]
    return scores


def bootstrap(differences, repetitions=10000, seed=1234):
    """Each row is one task's mean paired delta across seeds, three metrics."""
    import numpy as np
    rng = np.random.default_rng(seed)
    draws = {}
    for domain, rows in differences.items():
        rows = np.asarray(rows)
        indices = rng.integers(len(rows), size=(repetitions, len(rows)))
        draws[domain] = rows[indices].mean(axis=1)
    weights = {d: len(v) for d,v in differences.items()}
    draws['overall'] = sum(draws[d]*weights[d] for d in differences)/sum(weights.values())
    draws['macro'] = sum(draws[d] for d in differences)/len(differences)
    return {d: np.quantile(v, [0.025, 0.975], axis=0).T.tolist() for d,v in draws.items()}


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, help='Experiment batch directory')
    args = parser.parse_args()
    benchmark = args.root / 'benchmark/tau2/domains'
    task_ids = {}
    for domain, count in QUOTAS.items():
        if domain == 'banking_knowledge':
            ids = [t['id'] for t in json.loads((benchmark/domain/'tasks.json').read_text())]
        else:
            ids = json.loads((benchmark/domain/'split_tasks.json').read_text())['test']
        if len(ids) != count:
            raise ValueError(f'{domain}: wrong task quota')
        task_ids[domain] = ids
    data, diagnostics, table = {}, {}, []
    for arm in ARMS:
        data[arm], diagnostics[arm] = {}, {}
        for seed in (300,301):
            summary = json.loads((args.root/arm/'full/eval'/f'seed{seed}'/'summary.json').read_text())
            for key, expected in dict(seed=seed, num_trials=4, max_steps=200, max_errors=10,
                                      timeout=None, agent_eval_mode='official-native', agent='llm_agent',
                                      task_split_name='test', retrieval_config='bm25').items():
                if summary.get(key) != expected:
                    raise ValueError(f'{arm}/{seed}: unexpected {key}={summary.get(key)}')
            data[arm][seed], diagnostics[arm][seed] = {}, {}
            for domain in QUOTAS:
                ds = summary['domains'][domain]
                results = json.loads(Path(ds['results_file']).read_text())
                info = results.get('info') or {}
                agent = info.get('agent_info') or {}
                user = info.get('user_info') or {}
                for key, actual, expected in [
                    ('seed', info.get('seed'), seed),
                    ('trials', info.get('num_trials'), 4),
                    ('steps', info.get('max_steps'), 200),
                    ('agent', agent.get('implementation'), 'llm_agent'),
                    ('temperature', (agent.get('llm_args') or {}).get('temperature'), 0.6),
                    ('top_p', (agent.get('llm_args') or {}).get('top_p'), 1.0),
                    ('max_tokens', (agent.get('llm_args') or {}).get('max_tokens'), 1200),
                    ('user', user.get('llm'), 'openai/Qwen3.6-27B-tau2-user-nonthinking'),
                    ('user_temperature', (user.get('llm_args') or {}).get('temperature'), 0.0),
                    ('user_max_tokens', (user.get('llm_args') or {}).get('max_tokens'), 512),
                ]:
                    if actual != expected:
                        raise ValueError(f'{arm}/{seed}/{domain}: actual result {key}={actual}, expected {expected}')
                metrics = ds.get('metrics') or {}
                if metrics.get('infra_error_count', 0):
                    raise ValueError(f'{arm}/{seed}/{domain}: infrastructure errors')
                data[arm][seed][domain] = task_scores(results['simulations'], task_ids[domain])
                diagnostics[arm][seed][domain] = dict(
                    action_accuracy=(ds.get('diagnostics') or {}).get('action_accuracy'),
                    db_accuracy=(ds.get('diagnostics') or {}).get('db_accuracy'),
                    truncation=metrics.get('truncation_rate'),
                    infrastructure_errors=metrics.get('infra_error_count'),
                    termination_reasons=dict(Counter(s.get('termination_reason','unknown') for s in results['simulations'])))
        for seed in (300,301,'mean'):
            seeds = (300,301) if seed == 'mean' else (seed,)
            domain_rows = {d: np.mean([[data[arm][s][d][t] for t in sorted(task_ids[d])] for s in seeds],axis=0) for d in QUOTAS}
            values = {d: rows.mean(axis=0) for d,rows in domain_rows.items()}
            values['overall'] = np.concatenate(list(domain_rows.values())).mean(axis=0)
            values['macro'] = np.mean(list(values[d] for d in QUOTAS),axis=0)
            for domain, row in values.items():
                table.append(dict(arm=arm, seed=seed, domain=domain, **dict(zip(METRICS,row.tolist()))))
    comparisons = []
    pairs = [(a,'raw') for a in ARMS if a != 'raw'] + [(a,'mixed') for a in ARMS if a not in ('raw','mixed')]
    for arm, control in pairs:
        diff = {d: [np.mean([np.array(data[arm][s][d][t])-data[control][s][d][t] for s in (300,301)],axis=0).tolist()
                    for t in sorted(task_ids[d])] for d in QUOTAS}
        intervals = bootstrap(diff)
        point = {d: np.mean(v,axis=0).tolist() for d,v in diff.items()}
        point['overall'] = np.concatenate(list(diff.values())).mean(axis=0).tolist()
        point['macro'] = np.mean(list(point[d] for d in QUOTAS),axis=0).tolist()
        comparisons.append(dict(arm=arm, control=control, difference=point, ci95=intervals))
    payload = dict(table=table, comparisons=comparisons, diagnostics=diagnostics)
    (args.root/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines = ['# Four-domain SFT comparison', '', 'All values are percentages; seeds retain separate four-trial metrics.', '',
             '| Arm | Seed | Domain | pass@1 | pass@4(any) | pass^4 |', '|---|---|---|---:|---:|---:|']
    for row in table:
        lines.append('| '+ ' | '.join([row['arm'],str(row['seed']),row['domain']]+[f'{100*row[m]:.2f}' for m in METRICS])+' |')
    lines += ['', '## Paired differences', '', 'Percentage points; 95% task bootstrap CI, stratified by domain, paired across both seeds.', '',
              '| Arm − control | Domain | pass@1 | pass@4(any) | pass^4 |', '|---|---|---:|---:|---:|']
    for comparison in comparisons:
        for domain, values in comparison['difference'].items():
            cells = [f'{100*v:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]' for v,ci in zip(values, comparison['ci95'][domain])]
            lines.append('| '+' | '.join([comparison['arm']+' − '+comparison['control'],domain]+cells)+' |')
    lines += ['', 'Diagnostics (null = N/A) are in results.json. Compute differs between arms; one training seed; Banking retrieval distributions differ.']
    (args.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
