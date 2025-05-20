#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Entrypoint for all you evaluation-related needs.
Arguments are of the form key=value, and are passed to launch_evals.
Example usage:
python benchmark.py model_loader_kwargs.pretrained_weights=/path/to/weights.pth
"""

# Example command:
# python experiments/eval_dinov2_segmentation.py model_path=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/octo/eval/mae_loader.py model_loader_kwargs.model_name=vit_large_patch16 distributed=True ntasks_per_node=1 account=NAISS2025-5-255 gpus-per-node=A100:1 nodes=1 output_dir=./output_dir

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf
from dinov2.eval.segmentation.train_distributed import launch_jobs
from dinov2.eval.segmentation.utils import base_setup
from utils.cluster import get_slurm_partition

def launch_evals(
    output_dir: Path | str = ".",
    model_path: Path | str = Path(__file__).parent / "model.py",
    model_loader_kwargs: Mapping = {},
    evals_config_path: Path | str | None = Path(__file__).parent / "eval_config.yaml",
    evals_config: Mapping | None = None,
    distributed: bool = False,
    partition: str | None = None,
    pythonpath="",
    **sbatch_kwargs: Any,
):
    partition = partition or get_slurm_partition()

    output_dir = Path(output_dir).resolve()
    configs: Mapping[str, DictConfig] = OmegaConf.unsafe_merge(
        OmegaConf.load(evals_config_path) if evals_config_path is not None else {},
        evals_config if evals_config is not None else {},
    )  # pyright:ignore[reportAssignmentType]
    jobs = []
    for eval_name, eval_cfg in configs.items():
        eval_dir = output_dir / eval_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_cfg.model_path = str(model_path)
        eval_cfg.model_loader_kwargs = model_loader_kwargs
        eval_cfg.output_dir = eval_dir.as_posix()
        eval_cfg_path = eval_dir / "eval_config.yaml"
        if distributed:
            cmd = f"python -u {{codebase_dir}}/eval/{eval_cfg.pop('file')} {eval_cfg_path}"
        else:
            cmd = f"python eval/{eval_cfg.pop('file')} {eval_cfg_path}"
        OmegaConf.save(eval_cfg, eval_cfg_path)
        jobs.append((eval_name, cmd))
    if distributed:
        launch_jobs(
            output_dir,
            jobs,
            partition=partition,
            num_nodes=1,
            # ld_library_path=get_shared_libraries(),
            pythonpath=pythonpath,
            **sbatch_kwargs,
        )
    else:
        for job_name, cmd in jobs:
            print(f"Running {job_name}")
            print(f"cmd: {cmd}")
            subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    args = OmegaConf.from_cli()
    args.output_dir = './output_dir/'+args.model_loader_kwargs.model_name
    base_setup(args.output_dir, requeue=False, distributed=False)
    print(args)
    launch_evals(**args)  # type: ignore