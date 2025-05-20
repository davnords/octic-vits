import deit.main as training
from trainer import Trainer
from submit import get_args_parser, submit_jobs
from utils.cluster import get_shared_folder
import sys

huge_configs = {
    'batch': 64,
    'drop_path': 0.5,
    'nodes': 8,
    'ngpus': 4,
}

large_configs = {
    'batch': 128,
    'drop_path': 0.4,
    'nodes': 4,
    'ngpus': 4,
}

def main():
    description = "Submitit launcher for DeiT training"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    # Common configs:
    args.lr = 3e-3
    args.epochs = 400
    args.weight_decay = 0.02
    args.sched = "cosine"
    args.input_size = 224
    args.reprob = 0.0
    args.color_jitter = 0.3
    args.smoothing = 0.0
    args.warmup_epochs = 5
    args.drop = 0.0
    args.seed = 1337
    args.opt = "fusedlamb"
    args.warmup_lr = 1e-6
    args.mixup = 0.8
    args.cutmix = 1.0
    args.unscale_lr = True
    args.bce_loss = True
    args.ThreeAugment=True
    args.high_precision_matmul = True
    args.use_amp = True
    args.compile = True

    # Set configs based on model size
    if 'huge' in args.model:
        conf = huge_configs
    elif 'large' in args.model:
        conf = large_configs
    else:
        raise ValueError(f"Unknown model size: {args.model}")
    
    args.batch_size = conf['batch']
    args.drop_path = conf['drop_path']
    args.nodes = conf['nodes']
    args.ngpus = conf['ngpus']

    assert args.nodes * args.ngpus * args.batch_size == 2048, "Effective batch size should be 2048"

    args.output_dir = get_shared_folder() / "deit" / args.model / "%j"
    submit_jobs(Trainer, args, name=args.model)
    return 0

if __name__ == "__main__":
    sys.exit(main())