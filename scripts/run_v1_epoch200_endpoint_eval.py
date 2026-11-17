"""CLI entry point for V1 epoch-200 endpoint evaluation (Val94 or Val94 smoke subset)."""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from benchmark.endpoint_eval import (
        DEFAULT_NNUNET_CKPT_NAME,
        DEFAULT_NNUNET_FOLD,
        DEFAULT_NNUNET_LAST_CKPT,
        DEFAULT_NNUNET_MODEL_FOLDER,
        DEFAULT_RASNET_CKPT,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.lab.yaml")
    p.add_argument("--rasnet-ckpt", default=str(DEFAULT_RASNET_CKPT))
    p.add_argument("--nnunet-last-ckpt", default=str(DEFAULT_NNUNET_LAST_CKPT))
    p.add_argument("--nnunet-folder", default=str(DEFAULT_NNUNET_MODEL_FOLDER))
    p.add_argument("--nnunet-fold", type=int, default=DEFAULT_NNUNET_FOLD)
    p.add_argument("--nnunet-ckpt-name", default=DEFAULT_NNUNET_CKPT_NAME)
    p.add_argument("--out", "--out-dir", dest="out_dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--case-ids", nargs="+", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sw-batch-size", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-export-masks", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch
    from benchmark.endpoint_eval import run_endpoint_evaluation

    args = parse_args(argv)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    case_ids = args.case_ids
    if args.smoke and case_ids is None:
        case_ids = [14, 45]

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        out_dir = REPO_ROOT / "results" / f"v1_epoch200_smoke_{run_id}"
    else:
        out_dir = REPO_ROOT / "results" / f"v1_epoch200_comparison_{run_id}"

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    result_dir = run_endpoint_evaluation(
        repo_root=REPO_ROOT,
        config_path=Path(args.config),
        out_dir=out_dir,
        rasnet_ckpt=Path(args.rasnet_ckpt),
        nnunet_last_ckpt=Path(args.nnunet_last_ckpt),
        nnunet_model_folder=Path(args.nnunet_folder),
        nnunet_fold=args.nnunet_fold,
        nnunet_checkpoint_name=args.nnunet_ckpt_name,
        case_ids=case_ids,
        smoke=bool(args.smoke),
        device=args.device,
        sw_batch_size=args.sw_batch_size,
        use_amp=False if args.no_amp else None,
        export_masks=not args.no_export_masks,
        resume=bool(args.resume),
        run_id=run_id,
    )
    print(f"Endpoint evaluation complete: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
