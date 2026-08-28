"""Command line interface.

Every stage of the project is reachable as ``aerialdet <verb>`` so that the
README's commands are the same ones used to produce the numbers in it.
"""

from __future__ import annotations

import argparse
import json
import sys

from .paths import REPORTS_DIR


def _cmd_profile(args: argparse.Namespace) -> int:
    from ultralytics.data.utils import check_det_dataset

    from .paths import configure_ultralytics
    from .stats import profile_dataset, write_report

    configure_ultralytics()
    names = check_det_dataset(args.data)["names"]
    stats = profile_dataset(args.data)
    if not stats:
        print("No splits found. Run 'aerialdet download' first.", file=sys.stderr)
        return 1

    report = write_report(stats, names, REPORTS_DIR)
    for split, s in stats.items():
        print(
            f"{split:>5}: {s.n_images:,} images, {s.n_boxes:,} boxes, "
            f"{100 * s.small_fraction:.1f}% small"
        )
    print(f"\nWrote {report}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from ultralytics.data.utils import check_det_dataset

    from .paths import configure_ultralytics

    configure_ultralytics()
    spec = check_det_dataset(args.data)
    print(f"Dataset ready: {spec['nc']} classes at {spec['path']}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train

    result = train(args.config)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, write_comparison

    results = [
        evaluate(w, data=args.data, imgsz=args.imgsz, split=args.split, device=args.device)
        for w in args.weights
    ]
    report = write_comparison(results)
    print(report.read_text())
    return 0


def _cmd_tiled(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    from .tiling import predict_tiled

    model = YOLO(args.weights)
    out = predict_tiled(
        model,
        args.image,
        tile=args.tile,
        overlap=args.overlap,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    print(
        f"{len(out['boxes'])} detections across {out['n_tiles']} tiles "
        f"(tile={args.tile}, overlap={args.overlap})"
    )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export

    path = export(args.weights, fmt=args.format, imgsz=args.imgsz)
    print(f"Exported to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerialdet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch and convert the dataset")
    p.add_argument("--data", default="VisDrone.yaml")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("profile", help="measure class balance and object scale")
    p.add_argument("--data", default="VisDrone.yaml")
    p.set_defaults(func=_cmd_profile)

    p = sub.add_parser("train", help="fine-tune a model from a config")
    p.add_argument("config", help="config name (e.g. 'finetune_s') or path")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="validate one or more checkpoints")
    p.add_argument("weights", nargs="+")
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("tiled-predict", help="sliced inference on one image")
    p.add_argument("weights")
    p.add_argument("image")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_cmd_tiled)

    p = sub.add_parser("export", help="export a checkpoint for deployment")
    p.add_argument("weights")
    p.add_argument("--format", default="onnx")
    p.add_argument("--imgsz", type=int, default=960)
    p.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
