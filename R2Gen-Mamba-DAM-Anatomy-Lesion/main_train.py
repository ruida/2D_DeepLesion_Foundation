import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.trainer_deeplesion import DeepLesionTrainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from models.r2gen_deeplesion import R2GenDeepLesionModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/home/sun/data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/home/sun/data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray',
                        choices=['iu_xray', 'mimic_cxr', 'deeplesion'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # DeepLesion-specific settings
    parser.add_argument('--num_anatomy', type=int, default=174,
                        help='anatomy vocabulary size (max anatomy_id + 2); DeepLesion IDs go up to 172.')
    parser.add_argument('--max_boxes', type=int, default=2,
                        help='maximum number of bounding boxes per sample.')
    parser.add_argument('--max_anatomy', type=int, default=20,
                        help='maximum number of anatomy IDs per sample.')
    parser.add_argument('--bbox_hidden', type=int, default=128,
                        help='hidden dimension inside the bounding-box MLP encoder.')
    parser.add_argument('--anatomy_source', type=str, default='old',
                        choices=['old', 'rough', 'none'],
                        help=("DeepLesion anatomy input source: "
                              "old=original leakage-prone anatomy_ids; "
                              "rough=rough_anatomy_id(s)/rough_anatomy_name(s); "
                              "none=disable anatomy tokens."))
    parser.add_argument('--anatomy_encoding', type=str, default='id',
                        choices=['id', 'text'],
                        help=("How to encode anatomy guidance: "
                              "id=learned anatomy ID embedding; "
                              "text=tokenize anatomy names such as 'lesion lung' and embed them as text tokens."))

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

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the number of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the path to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the path to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loaders  (R2DataLoader now handles all three dataset_name values)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader   = R2DataLoader(args, tokenizer, split='val',   shuffle=False)
    test_dataloader  = R2DataLoader(args, tokenizer, split='test',  shuffle=False)

    criterion = compute_loss
    metrics   = compute_scores

    if args.dataset_name == 'deeplesion':
        # ---- DeepLesion: bbox + anatomy conditioning --------------------
        model        = R2GenDeepLesionModel(args, tokenizer)
        optimizer    = build_optimizer(args, model)
        lr_scheduler = build_lr_scheduler(args, optimizer)
        trainer = DeepLesionTrainer(
            model, criterion, metrics, optimizer, args, lr_scheduler,
            train_dataloader, val_dataloader, test_dataloader)
    else:
        # ---- Original path: IU X-Ray / MIMIC-CXR ------------------------
        model        = R2GenModel(args, tokenizer)
        optimizer    = build_optimizer(args, model)
        lr_scheduler = build_lr_scheduler(args, optimizer)
        trainer = Trainer(
            model, criterion, metrics, optimizer, args, lr_scheduler,
            train_dataloader, val_dataloader, test_dataloader)

    trainer.train()


if __name__ == '__main__':
    main()
