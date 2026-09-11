"""CPU checks for exact source subsets and actual batch coverage."""
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

NUM_GPUS = 0
SPEC = importlib.util.spec_from_file_location('prepare_domain_sft', Path(__file__).resolve().parents[1] / 'examples/tau2-bench/domain_generalization/prepare.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
SPEC2 = importlib.util.spec_from_file_location('summary_domain_sft', Path(SPEC.origin).with_name('summarize.py'))
summary = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(summary)


class PreparationTest(unittest.TestCase):
    def test_subsets_and_shuffle_preserve_rows(self):
        rows = [dict(domain=d, raw=str(i)) for d in prepare.COUNTS for i in range(20)]
        arms = prepare.partition(rows)
        self.assertEqual(arms, prepare.partition(rows))
        self.assertNotEqual(arms['mixed'], rows)
        self.assertCountEqual(arms['mixed'], rows)
        for domain in prepare.COUNTS:
            self.assertEqual(arms[domain], [r for r in rows if r['domain'] == domain])

    def test_target_mask_and_raw_preservation(self):
        row = dict(metadata=dict(domain='banking_knowledge'), tools=[], messages=[
            dict(role='assistant', content='old', step_loss_mask=0),
            dict(role='user', content='new'), dict(role='assistant', content='answer', step_loss_mask=1)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.jsonl'
            raw = json.dumps(row)
            path.write_text(raw + '\n')
            self.assertEqual(prepare.read_sources([path])[0]['raw'], raw)
            row['messages'][0]['step_loss_mask'] = 1
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, r'source.jsonl:1:'):
                prepare.read_sources([path])

    def test_continuous_batches_tail_coverage(self):
        stats = prepare.coverage([dict(tokens=10, supervised_tokens=2)] * 19)
        self.assertEqual(stats['updates'], 2)
        self.assertEqual(stats['coverage_counts'], {1: 6, 2: 13})
        self.assertEqual(stats['input_tokens'], 320)
        self.assertEqual(stats['supervised_tokens'], 64)


class MetricsTest(unittest.TestCase):
    def test_four_trials_and_missing_duplicate(self):
        rows = [dict(task_id='t', trial=i, reward_info=dict(reward=int(i == 0))) for i in range(4)]
        self.assertEqual(summary.task_scores(rows, ['t'])['t'], [0.25, 1, 0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summary.task_scores(rows + rows[:1], ['t'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            summary.task_scores(rows[:-1], ['t'])
        rows[0]['termination_reason'] = 'infrastructure_error'
        with self.assertRaisesRegex(ValueError, 'infrastructure'):
            summary.task_scores(rows, ['t'])

    def test_report_keeps_seed_trials_separate_and_task_weights(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy unavailable')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quotas = {'airline': 1, 'banking_knowledge': 2}
            for domain, count in quotas.items():
                folder = root / 'benchmark/tau2/domains' / domain
                folder.mkdir(parents=True)
                ids = [str(i) for i in range(count)]
                (folder / ('tasks.json' if domain == 'banking_knowledge' else 'split_tasks.json')).write_text(
                    json.dumps([{'id': t} for t in ids] if domain == 'banking_knowledge' else {'test': ids}))
            for arm in ('raw', 'mixed'):
                for seed in (300, 301):
                    folder = root / arm / 'full/eval' / f'seed{seed}'
                    folder.mkdir(parents=True)
                    doc = dict(seed=seed, num_trials=4, max_steps=200, max_errors=10, timeout=None,
                               agent_eval_mode='official-native', agent='llm_agent', task_split_name='test',
                               retrieval_config='bm25', domains={})
                    for domain, count in quotas.items():
                        # Opposite seeds cancel for paired comparisons; never collapse into eight trials.
                        reward = int((seed == 300) == (arm == 'mixed')) if domain == 'airline' else 0
                        simulations = [dict(task_id=str(t), trial=i, reward_info={'reward':reward}) for t in range(count) for i in range(4)]
                        info = dict(seed=seed, num_trials=4, max_steps=200,
                                    agent_info=dict(implementation='llm_agent',llm_args=dict(temperature=0.6,top_p=1.0,max_tokens=1200)),
                                    user_info=dict(llm='openai/Qwen3.6-27B-tau2-user-nonthinking',llm_args=dict(temperature=0.0,max_tokens=512)))
                        path = folder / f'{domain}.json'
                        path.write_text(json.dumps(dict(info=info, simulations=simulations)))
                        doc['domains'][domain] = dict(results_file=str(path),metrics=dict(infra_error_count=0))
                    (folder/'summary.json').write_text(json.dumps(doc))
            with mock.patch.object(summary, 'QUOTAS', quotas), mock.patch.object(summary, 'ARMS', ('raw','mixed')):
                with mock.patch('sys.argv', ['summarize.py', str(root)]):
                    summary.main()
            report = json.loads((root/'results.json').read_text())
            rows = [r for r in report['table'] if r['arm']=='mixed' and r['seed']=='mean']
            by_domain = {r['domain']:r for r in rows}
            self.assertEqual(by_domain['airline']['pass^4'], 0.5)
            self.assertAlmostEqual(by_domain['overall']['pass@1'], 1/6)
            self.assertEqual(by_domain['macro']['pass@1'], 0.25)
            self.assertEqual(report['comparisons'][0]['ci95']['overall'], [[0,0]]*3)

    def test_paired_bootstrap_constant_difference(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy unavailable')
        ci = summary.bootstrap({'a': [[0.25, 0, -1]] * 2, 'b': [[0.25, 0, -1]] * 3}, repetitions=100)
        self.assertEqual(ci['overall'], [[0.25, 0.25], [0, 0], [-1, -1]])


if __name__ == '__main__':
    unittest.main()
