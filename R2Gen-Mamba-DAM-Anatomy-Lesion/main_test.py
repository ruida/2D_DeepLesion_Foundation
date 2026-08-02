import argparse
import numpy as np
import torch

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester import Tester
from modules.tester_deeplesion import DeepLesionTester
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from models.r2gen_deeplesion import R2GenDeepLesionModel


def parse_args():
    parser = argparse.ArgumentParser()

    # -------------------------
    # Data input settings
    # -------------------------
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/iu_xray/images/",
        help="Path to image directory.",
    )
    parser.add_argument(
        "--ann_path",
        type=str,
        default="data/iu_xray/annotation.json",
        help="Path to annotation JSON.",
    )

    # -------------------------
    # Data loader settings
    # -------------------------
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="iu_xray",
        choices=["iu_xray", "mimic_cxr", "deeplesion"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=60,
        help="Maximum report sequence length.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Token frequency threshold.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size.",
    )

    # -------------------------
    # DeepLesion-specific settings
    # -------------------------
    parser.add_argument(
        "--num_anatomy",
        type=int,
        default=174,
        help="Anatomy vocabulary size. For rough anatomy, use 13.",
    )
    parser.add_argument(
        "--max_boxes",
        type=int,
        default=2,
        help="Maximum number of bbox slots.",
    )
    parser.add_argument(
        "--max_anatomy",
        type=int,
        default=20,
        help="Maximum number of anatomy tokens. For rough anatomy, use 1.",
    )
    parser.add_argument(
        "--bbox_hidden",
        type=int,
        default=128,
        help="Hidden dimension inside bounding-box MLP encoder.",
    )
    parser.add_argument(
        "--anatomy_source",
        type=str,
        default="old",
        choices=["old", "rough", "none"],
        help=(
            "DeepLesion anatomy source: "
            "old=original leakage-prone anatomy_ids; "
            "rough=rough_anatomy_id(s)/rough_anatomy_name(s); "
            "none=disable anatomy tokens."
        ),
    )
    parser.add_argument(
        "--anatomy_encoding",
        type=str,
        default="id",
        choices=["id", "text"],
        help=(
            "How to encode anatomy guidance: "
            "id=learned anatomy ID embedding; "
            "text=tokenize anatomy names such as 'lesion lung' and embed them as text tokens."
        ),
    )

    parser.add_argument('--include_lesion_type', action='store_true',
                        help='Append lesion_type_id/lesion_type_ids to the anatomy-conditioning token sequence.')

    # DAM global + focal image/mask settings
    parser.add_argument('--use_dam', action='store_true',
                        help='Use full RGB+mask and focal RGB+mask inputs with gated global-to-local attention.')
    parser.add_argument('--dam_image_size', type=int, default=224)
    parser.add_argument('--dam_crop_scale', type=float, default=3.0)
    parser.add_argument('--dam_min_crop_size', type=int, default=48)
    parser.add_argument('--dam_flip_prob', type=float, default=0.5)
    parser.add_argument('--bbox_format', type=str, default='xyxy', choices=['xyxy','xywh','yolo'])
    parser.add_argument('--dam_adapter_dim', type=int, default=512)
    parser.add_argument('--dam_num_heads', type=int, default=8)
    parser.add_argument('--dam_output_mode', type=str, default='local', choices=['local','concat'])

    # -------------------------
    # Visual extractor settings
    # -------------------------
    parser.add_argument(
        "--visual_extractor",
        type=str,
        default="resnet101",
        help="Visual extractor.",
    )
    parser.add_argument(
        "--visual_extractor_pretrained",
        type=bool,
        default=True,
        help="Whether to load pretrained visual extractor.",
    )

    # -------------------------
    # Encoder-decoder settings
    # -------------------------
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_vf", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--logit_layers", type=int, default=1)
    parser.add_argument("--bos_idx", type=int, default=0)
    parser.add_argument("--eos_idx", type=int, default=0)
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--use_bn", type=int, default=0)
    parser.add_argument("--drop_prob_lm", type=float, default=0.5)

    # Relational memory settings
    parser.add_argument("--rm_num_slots", type=int, default=3)
    parser.add_argument("--rm_num_heads", type=int, default=8)
    parser.add_argument("--rm_d_model", type=int, default=512)

    # -------------------------
    # Sampling settings
    # -------------------------
    parser.add_argument("--sample_method", type=str, default="beam_search")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_n", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--output_logsoftmax", type=int, default=1)
    parser.add_argument("--decoding_constraint", type=int, default=0)
    parser.add_argument("--block_trigrams", type=int, default=1)

    # -------------------------
    # Runtime / checkpoint settings
    # -------------------------
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="results/iu_xray")
    parser.add_argument("--record_dir", type=str, default="records/")
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--monitor_mode", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--monitor_metric", type=str, default="BLEU_4")
    parser.add_argument("--early_stop", type=int, default=50)

    # Optimization args kept for compatibility
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--lr_ve", type=float, default=5e-5)
    parser.add_argument("--lr_ed", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--amsgrad", type=bool, default=True)
    parser.add_argument("--lr_scheduler", type=str, default="StepLR")
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=9233)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Checkpoint path for testing. Prefer --load for main_test.py.",
    )

    args = parser.parse_args()

    # Backward compatibility:
    # If user passes --resume instead of --load, use it as load checkpoint.
    if args.load is None and args.resume is not None:
        args.load = args.resume

    if args.load is None:
        raise ValueError(
            "Testing requires a checkpoint. Please pass "
            "--load results/.../model_best.pth "
            "or --resume results/.../model_best.pth"
        )

    return args


def main():
    args = parse_args()

    print("Testing configuration:")
    print("  dataset_name    :", args.dataset_name)
    print("  image_dir       :", args.image_dir)
    print("  ann_path        :", args.ann_path)
    print("  save_dir        :", args.save_dir)
    print("  load checkpoint :", args.load)
    print("  anatomy_source  :", args.anatomy_source)
    print("  anatomy_encoding:", args.anatomy_encoding)
    print("  num_anatomy     :", args.num_anatomy)
    print("  max_anatomy     :", args.max_anatomy)
    print("  max_boxes       :", args.max_boxes)
    print("  use_dam         :", args.use_dam)
    print("  include_lesion_type:", args.include_lesion_type)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    # IMPORTANT:
    # Use R2DataLoader, because it already has collate_fn_deeplesion.
    # Do not use default PyTorch DataLoader for DeepLesion.
    test_dataloader = R2DataLoader(
        args,
        tokenizer,
        split="test",
        shuffle=False,
    )

    criterion = compute_loss
    metrics = compute_scores

    if args.dataset_name == "deeplesion":
        model = R2GenDeepLesionModel(args, tokenizer)
        tester = DeepLesionTester(
            model,
            criterion,
            metrics,
            args,
            test_dataloader,
        )
    else:
        model = R2GenModel(args, tokenizer)
        tester = Tester(
            model,
            criterion,
            metrics,
            args,
            test_dataloader,
        )

    tester.test()


if __name__ == "__main__":
    main()
