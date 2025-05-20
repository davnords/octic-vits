import dinov2.eval.linear as training
from trainer import Trainer
from submit import get_args_parser, submit_jobs
import sys

# example command: python experiments/eval_dinov2.py <path_to_training> 12499

def main():
    for eval in ['linear', 'knn']:
        if eval == 'linear':
            import dinov2.eval.linear as training
            description = "Submitit launcher for DINOv2 linear evaluation"
        elif eval == 'knn':
            import dinov2.eval.knn as training
            description = "Submitit launcher for DINOv2 knn evaluation"

        train_args_parser = training.get_args_parser()
    
        parents = [train_args_parser]
        args_parser = get_args_parser(description=description, parents=parents, add_help=False)
        args = args_parser.parse_args()
        args.training_module = training.__name__

        args.timeout = 150      
        args.output_dir = args.dir + f"/eval/training_{args.ckpt_iter}/{eval}"
        print(f"Submitted {eval} evaluation for iteration {args.ckpt_iter} to {args.output_dir}")

        submit_jobs(Trainer, args, name='dinov2_eval')
    return 0

if __name__ == "__main__":
    sys.exit(main())