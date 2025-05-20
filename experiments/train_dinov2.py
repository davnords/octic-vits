import dinov2.train.train as training
from trainer import Trainer
from submit import get_args_parser, submit_jobs
from utils.cluster import get_shared_folder
from dinov2.logging import setup_logging
import sys
import yaml
import os 
import constants

# Example command: python experiments/train_dinov2.py --config-file dinov2/configs/train/hybrid_vith16.yaml --ngpus 4 --nodes 2

os.environ["IMAGENET_PATH"] = constants.IMAGENET_PATH
os.environ["EXTRA_PATH"] = constants.EXTRA_PATH

def main():
    description = "Submitit launcher for DINOv2 training"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents, add_help=False)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    setup_logging()

    model = yaml.safe_load(open(args.config_file))['student']['arch']

    args.output_dir = get_shared_folder() / "dinov2" / model / "%j"
    submit_jobs(Trainer, args, name='dinov2:train')
    return 0

if __name__ == "__main__":
    sys.exit(main())