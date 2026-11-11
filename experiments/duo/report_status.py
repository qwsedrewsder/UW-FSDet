import json
import re
from pathlib import Path

root = Path('/root/duo_exp')
state = json.loads((root / 'queue_state.json').read_text())
done = [name for name in state.get('done', []) if not name.startswith('e2_')]
schedule = json.loads((root / 'schedule.json').read_text())['jobs']
reported = [job for job in schedule if not (job['kind'] == 'htdet' and job['model'] == 'e2')]
print(f"Completed reported runs: {len(done)}/{len(reported)}")
print(f"Queue: {state.get('phase')} | current: {state.get('current', 'none')}")
print('Model\tSeed\tAP\tAP50\tAP75\tAPs\tAPm\tAPl')
for name in done:
    result = root / 'runs' / name / 'test_eval.json'
    if not result.exists():
        continue
    metrics = json.loads(result.read_text())
    model, seed = name.rsplit('_seed', 1)
    values = [metrics[k] * 100 for k in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')]
    print(model + '\t' + seed + '\t' + '\t'.join(f'{value:.2f}' for value in values))
if state.get('current') == 'hyuod_seed0':
    log = (root / 'logs/hyuod_seed0.queue.log').read_text(errors='replace')
    epochs = re.findall(r'\b(\d+)/400\b', log)
    if epochs:
        print(f"HyUOD seed 0 epoch: {epochs[-1]}/400")
