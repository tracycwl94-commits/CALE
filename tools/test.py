"""Evaluate a frozen detector with the supplied baseline or CALE configuration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-root', help='LVIS or V3Det dataset root')
    parser.add_argument('--work-dir')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0,
                        help='Smoke check on the first N images; saves predictions without AP evaluation')
    args = parser.parse_args()
    config, checkpoint = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    data_root = str(Path(args.data_root).resolve()) if args.data_root else None
    work_dir = str(Path(args.work_dir).resolve()) if args.work_dir else None
    os.chdir(ROOT)
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    register_all_modules()
    cfg = Config.fromfile(str(config))
    dataset = cfg.test_dataloader.dataset
    if data_root:
        dataset.data_root = data_root
        cfg.test_evaluator.ann_file = str(Path(data_root) / dataset.ann_file)
    if dataset.type == 'V3DetDataset':
        dataset.label_file = str(ROOT / 'stat_files/v3det_categories.txt')
    if args.batch_size:
        cfg.test_dataloader.batch_size = args.batch_size
    cfg.test_dataloader.num_workers = args.workers
    cfg.test_dataloader.persistent_workers = args.workers > 0
    cfg.work_dir = work_dir or cfg.work_dir
    output = Path(cfg.work_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(output)
    cfg.test_evaluator.outfile_prefix = str(output / 'predictions')
    is_v3det = dataset.type == 'V3DetDataset'
    ann_file = cfg.test_evaluator.ann_file
    if args.limit:
        dataset.indices = args.limit
        cfg.test_evaluator = dict(type='DumpDetResults', out_file_path=str(output / 'smoke.pkl'))
    cfg.load_from = None
    runner = Runner.from_cfg(cfg)
    # Checkpoints from the linked original projects contain trusted metadata.
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    state = state.get('state_dict', state)
    if all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    runner.model.load_state_dict(state, strict=True)
    del state
    metrics = runner.test()
    if args.limit:
        print(f'Smoke inference complete ({args.limit} images); no AP was computed.')
    elif is_v3det:
        # Match the paper's category-wise COCO bbox protocol, maxDets=300.
        subprocess.run([sys.executable, str(ROOT / 'tools/evaluate_v3det.py'),
                        '--annotations', ann_file,
                        '--predictions', str(output / 'predictions.bbox.json'),
                        '--groups', str(ROOT / 'stat_files/v3det_groups.json'),
                        '--output', str(output / 'metrics.json')], check=True)
    else:
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
