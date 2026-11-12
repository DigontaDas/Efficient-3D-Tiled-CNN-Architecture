"""D9 diagnostic: is a V-Net pilot collapse a BatchNorm running-statistics artefact?

Validation (val20) cases only; writes Rasnet-v2/results/diagnostics/vnet_bn_diag_<run>.csv.
Never used for LR, design, threshold or checkpoint selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))

from augmentations import build_raw_eval_transform  # noqa: E402
from dataset_pairs import resolve_case_pairs  # noqa: E402
from eval_fair_v2 import build_monai_invert_op, compute_metrics, infer_monai_native_zyx  # noqa: E402
from rasnet_v2_model import build_monai_model  # noqa: E402

_DROPOUTS = (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)


class BatchStatsBN(torch.nn.Module):
    """Wrapper whose eval() keeps BatchNorm on batch statistics while dropout stays off."""

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)

    def train(self, mode: bool = True):
        self.training = mode
        self.inner.train(True)
        for m in self.inner.modules():
            if isinstance(m, _DROPOUTS):
                m.eval()
        return self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(V2 / "configs" / "fair_v2.yaml"))
    ap.add_argument("--run", default="pilot2a_vnet_f16_lr1e-4_seed42")
    ap.add_argument("--ckpt", default="epoch_50.pth")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ids = json.loads(Path(cfg["study"]["val20_ids_file"]).read_text(encoding="utf-8"))["case_ids"][: args.n_cases]
    if any(851 <= int(c) <= 1000 for c in ids):
        raise ValueError("Locked test IDs in diagnostic case list")
    pairs = resolve_case_pairs(cfg["study"]["dataset_root"], [int(c) for c in ids], check_headers=False)

    dev = torch.device(args.device)
    net = build_monai_model(arch="vnet", init_filters=16, gate_mode="conv_bias", enable_ds=True).to(dev)
    ckpt = Path(cfg["study"]["root_dir"]) / "checkpoints" / args.run / args.ckpt
    obj = torch.load(ckpt, map_location=dev, weights_only=True)
    net.load_state_dict(obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj,
                        strict=True)
    net.eval()
    bs_net = BatchStatsBN(copy.deepcopy(net)).to(dev)

    tf = build_raw_eval_transform(pixdim_xyz=tuple(cfg["preprocessing"]["pixdim_xyz"]),
                                  body_crop_hu=float(cfg["preprocessing"]["body_crop_hu"]))
    inv = build_monai_invert_op(tf)
    ie = cfg["inference_eval"]
    rows = []
    for cid, (img_p, lbl_p) in pairs.items():
        lbl = sitk.ReadImage(str(lbl_p))
        gt = sitk.GetArrayFromImage(lbl) > 0
        sp = tuple(float(s) for s in reversed(lbl.GetSpacing()))
        for mode, model in (("bn_running_stats", net), ("bn_batch_stats", bs_net)):
            prob, _ = infer_monai_native_zyx(model, img_p, tf, roi_size_xyz=tuple(ie["roi_size_xyz"]),
                                             sw_overlap=float(ie["sw_overlap"]), sw_mode=str(ie["sw_mode"]),
                                             sw_batch_size=int(ie["sw_batch_size"]), need_tta=False,
                                             device=str(dev), inv_op=inv)
            m = compute_metrics((prob >= 0.5).astype(np.uint8), gt, sp)
            rows.append({"run": args.run, "ckpt": args.ckpt, "case_id": int(cid), "bn_mode": mode,
                         "dice": m["dice"], "precision": m["precision"], "status": m["status"],
                         "fg_prob_max": float(prob.max())})
            print(rows[-1], flush=True)

    out = V2 / "results" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"vnet_bn_diag_{args.run}.csv", index=False)
    print(df.groupby("bn_mode")[["dice", "precision", "fg_prob_max"]].mean())


if __name__ == "__main__":
    main()
