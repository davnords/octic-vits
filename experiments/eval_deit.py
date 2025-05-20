import deit.main as training
from trainer import Trainer
from submit import get_args_parser, submit_jobs
from utils.cluster import get_shared_folder
import sys

def main():
    description = "Submitit launcher for DeiT evaluation"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    args.output_dir = get_shared_folder() / "deit" / args.model / "%j"
    submit_jobs(Trainer, args, name=args.model)
    return 0

if __name__ == "__main__":
    sys.exit(main())