"""Check this experiment every 30 minutes and fill at most two eval slots."""

import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
JOB = '/mnt/afs/users/fush/job'
BUDGETS = {'airline': 60, 'retail': 30, 'telecom': 20, 'banking': 30}
TOTALS = {'airline': 80, 'retail': 160, 'telecom': 160, 'banking_knowledge': 388}
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED', 'CANCELLED'}


def show(job_id):
    output = subprocess.check_output([JOB, 'show', str(job_id)], text=True, timeout=60)
    return json.JSONDecoder().raw_decode(output.lstrip())[0]


def report(message):
    print(message, flush=True)
    with (ROOT / 'README.md').open('a') as output:
        output.write(message + '\n\n')


def check():
    now = datetime.datetime.now()
    report(f'Supervision {now:%Y-%m-%d %H:%M:%S} CST.')
    train_jobs = [show(22033), show(22034)]
    checkpoints = []
    progress = []
    for domain, budget in BUDGETS.items():
        run = ROOT / 'arms' / 'async' / f'20260920_sft4505-{domain}-train'
        log = run / 'run.log'
        text = log.read_text(errors='replace').replace('\x00', '') if log.exists() else ''
        batches = re.findall(r'tau2_trainer batch=(\d+) start=([\d.]+) end=([\d.]+)', text)
        completed = len({batch[0] for batch in batches})
        progress.append(f'{domain} {completed}/{budget}')
        marker = run / 'checkpoints' / 'latest_checkpointed_iteration.txt'
        if marker.exists():
            latest = int(marker.read_text().strip())
            for path in marker.parent.glob('iter_*'):
                match = re.fullmatch(r'iter_(\d+)', path.name)
                if match and int(match[1]) <= latest:
                    checkpoints.append((path.stat().st_mtime, domain, int(match[1])))
    report('Training: ' + ', '.join(progress) + '. Jobs: ' +
           ', '.join(f'{j["id"]} {j["state"]}' for j in train_jobs) + '.')

    evaluations = {}
    attention = []
    active = 0
    for directory in sorted((ROOT / 'jobs').iterdir()):
        match = re.match(r'(\d+)-sft4505-(airline|retail|telecom|banking)-iter(\d+)-four-domain-', directory.name)
        if not match:
            continue
        job_id, domain, iteration = match.groups()
        key = (domain, int(iteration))
        job = show(job_id)
        evaluations[key] = job
        if job['state'] not in TERMINAL:
            active += 1
        if job['state'] in {'FAILED', 'STOPPED', 'CANCELLED'} and int(job_id) != 22077:
            attention.append(f'Evaluation {job_id} is {job["state"]}; inspect before retry.')
        counts = {}
        result_files = {}
        eval_dir = ROOT / 'eval' / f'{domain}-iter{iteration}-four-domain'
        for path in eval_dir.glob('trajectories/*/results.json'):
            for eval_domain in TOTALS:
                if f'_{eval_domain}_test_' in path.parent.name:
                    try:
                        data = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        print(f'Result being written; retry next observation: {path}', flush=True)
                        continue
                    counts[eval_domain] = len(data.get('simulations', []))
                    result_files[eval_domain] = path
        report(f'Eval {job_id} {domain} iter{iteration}: {job["state"]}; ' +
               ', '.join(f'{d} {counts.get(d, 0)}/{n}' for d, n in TOTALS.items()) + '.')
        banking = result_files.get('banking_knowledge')
        if job['state'] == 'RUNNING' and banking and counts['banking_knowledge'] < 388:
            idle_minutes = (time.time() - banking.stat().st_mtime) / 60
            if idle_minutes >= 60:
                subprocess.run([JOB, 'stop', job_id], check=True, timeout=60)
                attention.append(f'Stopped {job_id}: Banking had no saved result for {idle_minutes:.1f} minutes. Keep partial results;omit Banking from subsequent evaluations before resuming submissions.')
        if job['state'] == 'SUCCEEDED' and any(counts.get(d, 0) != n for d, n in TOTALS.items()):
            attention.append(f'Evaluation {job_id} exited successfully but domain counts are incomplete.')
    for job in train_jobs:
        if job['state'] in {'FAILED', 'STOPPED', 'CANCELLED'}:
            attention.append(f'Training {job["id"]} is {job["state"]}.')
    if attention:
        report('Attention required: ' + ' '.join(attention))
        return True

    for _, domain, iteration in sorted(checkpoints):
        if active >= 2:
            break
        if (domain, iteration) in evaluations:
            continue
        command = ['bash', 'scripts/submit.sh', '--experiment', ROOT.name,
                   '--gpus', '8', '--name', f'sft4505-{domain}-iter{iteration}-four-domain',
                   str(ROOT / 'eval_checkpoint.sh'), '--', domain, str(iteration)]
        output = subprocess.check_output(command, cwd=PROJECT, text=True, timeout=120)
        print(output, flush=True)
        job_id = int(re.search(r'\bid=(\d+)', output)[1])
        job = show(job_id)
        run_log = Path(job['run_log'].replace('run_*_', 'run_0_'))
        report(f'Evaluation job{job_id}: {domain} iter{iteration},8 GPUs,unchanged full four-domain protocol. '
               f'[Run log]({run_log.relative_to(ROOT)}).')
        evaluations[(domain, iteration)] = job
        active += 1
    expected = {(d, i) for d, n in BUDGETS.items() for i in range(9, n, 10)}
    if all(j['state'] == 'SUCCEEDED' for j in train_jobs) and all(
        key in evaluations and (evaluations[key]['state'] == 'SUCCEEDED' or evaluations[key]['id'] == 22077) for key in expected
    ):
        report('All four expert training jobs and all14 checkpoint evaluation attempts finished. Banking iter19 job22077 has one context overflow counted as task failure per user instruction;review summaries before closing the goal.')
        return True
    return False


if __name__ == '__main__':
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
    while True:
        if check():
            break
        time.sleep(1800)
