# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import datetime
import time
import math
import json
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from dense_diagnostics import compute_dense_diagnostics, save_attention_maps
from clean_horizon_contract import (
    CleanHorizonContractError,
    build_training_contract,
    capture_rank_rng_states,
    accumulation_group_size,
    create_amp_overflow_state,
    get_git_state,
    record_amp_optimizer_attempt,
    restore_rank_rng_state,
    optimizer_steps_per_epoch,
    should_stop_before_next_epoch,
    validate_amp_overflow_state,
    validate_milestone_epochs,
    validate_resume_checkpoint,
    write_json_atomic,
)

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    # Removed torch.hub.list("facebookresearch/xcit:main") to prevent DDP download race condition
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--amp_max_consecutive_overflows', type=int, default=3,
        help='Abort after this many consecutive GradScaler optimizer-step skips.')
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--resume_from', default='', type=str,
        help='Path to a checkpoint to resume from. Defaults to output_dir/checkpoint.pth when empty.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Dense Degradation diagnostics
    parser.add_argument('--val_data_path', default='', type=str,
        help='Path to validation data for dense degradation diagnostics.')
    parser.add_argument('--diag_every', default=10, type=int,
        help='Compute dense degradation diagnostics every N epochs.')
    parser.add_argument('--attn_viz_every', default=50, type=int,
        help='Save attention map visualizations every N epochs.')
    parser.add_argument('--diag_num_batches', default=50, type=int,
        help='Number of validation batches to use for diagnostics.')

    # Gradient accumulation (for small-GPU training)
    parser.add_argument('--accum_steps', default=1, type=int,
        help='Gradient accumulation steps. Effective batch = batch_size_per_gpu * accum_steps.')
    parser.add_argument('--drop_incomplete_accumulation', default=False, type=utils.bool_flag,
        help='Drop a final partial accumulation group to keep every optimizer batch equal.')

    # Checkpoint management
    parser.add_argument('--keep_last_ckpts', default=0, type=int,
        help='Keep only the last N periodic checkpoints (0 = keep all).')
    parser.add_argument('--milestone_ckpt_epochs', default=(), type=int, nargs='*',
        help='Always save these zero-based epoch labels in addition to saveckp_freq.')
    parser.add_argument('--strict_resume_schedule', default=False, type=utils.bool_flag,
        help='Fail before training if resume metadata differs from this launch.')
    parser.add_argument('--expected_world_size', default=0, type=int,
        help='Required distributed world size when non-zero.')
    parser.add_argument('--max_runtime_hours', default=0.0, type=float,
        help='Session wall-clock budget; zero disables the epoch-boundary guard.')
    parser.add_argument('--runtime_reserve_minutes', default=0.0, type=float,
        help='Time reserved for checkpoint publication before the platform limit.')
    parser.add_argument('--run_name', default='', type=str,
        help='Stable experiment identifier recorded in the training contract.')
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    validate_milestone_epochs(args)
    if args.expected_world_size and utils.get_world_size() != args.expected_world_size:
        raise CleanHorizonContractError(
            f"Expected world size {args.expected_world_size}, got {utils.get_world_size()}"
        )
    source_state = get_git_state(Path(__file__).resolve().parent)
    if args.strict_resume_schedule and source_state["source_dirty"]:
        raise CleanHorizonContractError(
            "Strict clean-horizon training requires a clean source checkout"
        )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    amp_overflow_state = create_amp_overflow_state(
        args.amp_max_consecutive_overflows
    )

    # ============ init schedulers ... ============
    # When using gradient accumulation, the effective number of optimizer steps
    # per epoch is reduced. We scale the schedules accordingly so that the
    # total learning rate / momentum trajectory stays the same.
    effective_loader_len = optimizer_steps_per_epoch(
        len(data_loader),
        args.accum_steps,
        drop_incomplete=args.drop_incomplete_accumulation,
    )
    if effective_loader_len <= 0:
        raise CleanHorizonContractError(
            "The data loader does not provide one optimizer step per epoch"
        )
    training_contract = build_training_contract(
        args,
        dataset_size=len(dataset),
        class_count=len(dataset.classes),
        batches_per_epoch=len(data_loader),
        optimizer_steps_per_epoch=effective_loader_len,
        world_size=utils.get_world_size(),
        source_state=source_state,
    )
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * args.accum_steps * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, effective_loader_len,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, effective_loader_len,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, effective_loader_len)
    print(f"Loss, optimizer and schedulers ready.")
    print(f"Effective batch size: {args.batch_size_per_gpu * args.accum_steps * utils.get_world_size()}")
    print(f"Gradient accumulation steps: {args.accum_steps}")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    resume_path = args.resume_from or os.path.join(args.output_dir, "checkpoint.pth")
    resume_identity = None
    if args.strict_resume_schedule:
        if args.resume_from and not os.path.isfile(resume_path):
            raise CleanHorizonContractError(
                f"Explicit resume checkpoint does not exist: {resume_path}"
            )
        if os.path.isfile(resume_path):
            resume_identity = validate_resume_checkpoint(
                resume_path,
                training_contract,
                use_fp16=args.use_fp16,
            )
    restored_checkpoint = utils.restart_from_checkpoint(
        resume_path,
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    if args.strict_resume_schedule and resume_identity is not None:
        if restored_checkpoint is None:
            raise CleanHorizonContractError("Validated checkpoint was not restored")
        if start_epoch != resume_identity["completed_epochs"]:
            raise CleanHorizonContractError(
                "Restored epoch does not match the validated checkpoint identity"
            )
        restore_rank_rng_state(restored_checkpoint["rng_states"], utils.get_rank())
    if restored_checkpoint is not None and "amp_overflow_state" in restored_checkpoint:
        amp_overflow_state = validate_amp_overflow_state(
            restored_checkpoint["amp_overflow_state"],
            expected_max_consecutive_overflows=args.amp_max_consecutive_overflows,
        )
    del restored_checkpoint

    start_time = time.time()
    session_epoch_durations = []
    final_status = "complete" if start_epoch >= args.epochs else "running"
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        epoch_started_at = time.time()
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, amp_overflow_state, args)

        # ============ dense degradation diagnostics ... ============
        if utils.is_main_process() and args.val_data_path and epoch % args.diag_every == 0:
            print(f"Computing dense degradation diagnostics at epoch {epoch}...")
            # Use teacher backbone (without projection head) for feature extraction
            backbone = teacher_without_ddp.backbone
            diag_stats = compute_dense_diagnostics(
                backbone, args.val_data_path, torch.device('cuda'),
                num_batches=args.diag_num_batches,
            )
            train_stats.update(diag_stats)
            print(f"  effective_rank={diag_stats.get('diag_effective_rank', 'N/A'):.2f}  "
                  f"cls_patch_cos={diag_stats.get('diag_cls_patch_cosine', 'N/A'):.4f}  "
                  f"cond_number={diag_stats.get('diag_condition_number', 'N/A'):.2f}")

        # ============ attention map visualization ... ============
        if utils.is_main_process() and args.val_data_path and epoch % args.attn_viz_every == 0:
            print(f"Saving attention maps at epoch {epoch}...")
            backbone = teacher_without_ddp.backbone
            save_attention_maps(
                backbone, args.val_data_path, epoch, args.output_dir,
                torch.device('cuda'),
            )

        # ============ writing logs ... ============
        rng_states = capture_rank_rng_states()
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            'training_contract': training_contract,
            'rng_states': rng_states,
            'amp_overflow_state': dict(amp_overflow_state),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        save_periodic = bool(args.saveckp_freq and epoch % args.saveckp_freq == 0)
        save_milestone = epoch in set(args.milestone_ckpt_epochs)
        if save_periodic or save_milestone:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        # ============ cleanup old checkpoints ... ============
        if utils.is_main_process() and args.keep_last_ckpts > 0:
            periodic_ckpts = sorted(glob(os.path.join(args.output_dir, 'checkpoint[0-9]*.pth')))
            for old_ckpt in periodic_ckpts[:-args.keep_last_ckpts]:
                os.remove(old_ckpt)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        completed_epochs = epoch + 1
        session_epoch_durations.append(time.time() - epoch_started_at)
        elapsed_seconds = time.time() - start_time
        mean_epoch_seconds = sum(session_epoch_durations) / len(session_epoch_durations)
        stop_for_runtime = False
        if utils.is_main_process():
            stop_for_runtime = should_stop_before_next_epoch(
                elapsed_seconds=elapsed_seconds,
                mean_epoch_seconds=mean_epoch_seconds,
                max_runtime_seconds=args.max_runtime_hours * 3600.0,
                reserve_seconds=args.runtime_reserve_minutes * 60.0,
                completed_epochs=completed_epochs,
                target_epochs=args.epochs,
            )
        if dist.is_available() and dist.is_initialized():
            stop_tensor = torch.tensor(
                [int(stop_for_runtime)],
                device=torch.device("cuda", args.gpu),
                dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            stop_for_runtime = bool(stop_tensor.item())

        if completed_epochs >= args.epochs:
            final_status = "complete"
        elif stop_for_runtime:
            final_status = "partial_runtime_guard"
        else:
            final_status = "running"
        if utils.is_main_process():
            rolling_path = Path(args.output_dir) / "checkpoint.pth"
            write_json_atomic(
                Path(args.output_dir) / "clean_horizon_session_summary.json",
                {
                    "status": final_status,
                    "run_name": args.run_name,
                    "session_start_epoch": start_epoch,
                    "completed_epochs": completed_epochs,
                    "last_epoch_label": epoch,
                    "target_epochs": args.epochs,
                    "session_elapsed_seconds": elapsed_seconds,
                    "mean_epoch_seconds": mean_epoch_seconds,
                    "resume_checkpoint": resume_identity,
                    "rolling_checkpoint": {
                        "basename": rolling_path.name,
                        "size_bytes": rolling_path.stat().st_size,
                        "completed_epochs": completed_epochs,
                    },
                    "training_contract": training_contract,
                    "amp_overflow_state": dict(amp_overflow_state),
                    "amp_scale": (
                        float(fp16_scaler.get_scale())
                        if fp16_scaler is not None
                        else None
                    ),
                },
            )
        if stop_for_runtime:
            print(
                "Runtime guard stopped after completed epoch "
                f"{completed_epochs}; checkpoint and session summary are ready."
            )
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {} (status={})'.format(total_time_str, final_status))


def append_clean_horizon_event(args, payload):
    """Append one rank-zero event so recoverable skips remain auditable."""
    if not args.strict_resume_schedule or not utils.is_main_process():
        return
    event = {
        "run_name": args.run_name,
        **payload,
    }
    with (Path(args.output_dir) / "clean_horizon_events.jsonl").open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, amp_overflow_state, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    accum_steps = args.accum_steps
    optimizer_steps = optimizer_steps_per_epoch(
        len(data_loader),
        accum_steps,
        drop_incomplete=args.drop_incomplete_accumulation,
    )
    usable_batches = (
        optimizer_steps * accum_steps
        if args.drop_incomplete_accumulation
        else len(data_loader)
    )
    epoch_overflows_before = amp_overflow_state["total_overflows"]
    epoch_attempts_before = amp_overflow_state["optimizer_step_attempts"]
    epoch_applied_before = amp_overflow_state["optimizer_steps_applied"]
    optimizer.zero_grad()
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if it >= usable_batches:
            break
        # update weight decay and learning rate according to their schedule
        # With gradient accumulation, we update LR/WD every accum_steps
        opt_step = optimizer_steps * epoch + it // accum_steps
        opt_step = min(opt_step, len(lr_schedule) - 1)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[opt_step]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[opt_step]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        nonfinite_loss = torch.tensor(
            [int(not math.isfinite(loss.item()))],
            device=torch.device("cuda", args.gpu),
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(nonfinite_loss, op=dist.ReduceOp.MAX)
        if nonfinite_loss.item():
            append_clean_horizon_event(args, {
                "event": "nonfinite_loss",
                "epoch": epoch,
                "iteration": it,
                "loss": float(loss.item()),
            })
            raise FloatingPointError(
                "Non-finite DINO loss at epoch {} iteration {}: {}".format(
                    epoch, it, loss.item()
                )
            )

        # Scale loss for gradient accumulation
        group_size = accumulation_group_size(
            it,
            len(data_loader),
            accum_steps,
            drop_incomplete=args.drop_incomplete_accumulation,
        )
        loss = loss / group_size

        # backward pass (accumulate gradients)
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()

        # optimizer step only every accum_steps iterations
        if (it + 1) % accum_steps == 0 or (it + 1) == usable_batches:
            optimizer_stepped = True
            if fp16_scaler is None:
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args.freeze_last_layer)
                optimizer.step()
                record_amp_optimizer_attempt(
                    amp_overflow_state,
                    overflowed=False,
                )
            else:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args.freeze_last_layer)
                scale_before = fp16_scaler.get_scale()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer_stepped = fp16_scaler.get_scale() >= scale_before
                overflow_count = torch.tensor(
                    [int(not optimizer_stepped)],
                    device=torch.device("cuda", args.gpu),
                    dtype=torch.int32,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(overflow_count, op=dist.ReduceOp.SUM)
                overflow_ranks = int(overflow_count.item())
                world_size = utils.get_world_size()
                if overflow_ranks not in (0, world_size):
                    append_clean_horizon_event(args, {
                        "event": "amp_overflow_rank_mismatch",
                        "epoch": epoch,
                        "iteration": it,
                        "optimizer_step_slot": opt_step,
                        "overflow_ranks": overflow_ranks,
                        "world_size": world_size,
                    })
                    raise FloatingPointError(
                        "AMP overflow decision differed across ranks at epoch "
                        f"{epoch} iteration {it}: {overflow_ranks}/{world_size} ranks skipped"
                    )
                optimizer_stepped = overflow_ranks == 0
                limit_reached = record_amp_optimizer_attempt(
                    amp_overflow_state,
                    overflowed=not optimizer_stepped,
                )
                if not optimizer_stepped:
                    append_clean_horizon_event(args, {
                        "event": (
                            "amp_overflow_kill_limit"
                            if limit_reached
                            else "amp_overflow_recovered"
                        ),
                        "epoch": epoch,
                        "iteration": it,
                        "optimizer_step_slot": opt_step,
                        "scale_before": float(scale_before),
                        "scale_after": float(fp16_scaler.get_scale()),
                        "amp_overflow_state": dict(amp_overflow_state),
                    })
                if limit_reached:
                    raise FloatingPointError(
                        "AMP overflow reached the consecutive skip limit at epoch "
                        f"{epoch} iteration {it}: "
                        f"{amp_overflow_state['consecutive_overflows']}/"
                        f"{amp_overflow_state['max_consecutive_overflows']} consecutive, "
                        f"{amp_overflow_state['total_overflows']} total"
                    )
            optimizer.zero_grad()

            # EMA update for the teacher (only at optimizer steps)
            if optimizer_stepped:
                with torch.no_grad():
                    m_step = min(opt_step, len(momentum_schedule) - 1)
                    m = momentum_schedule[m_step]
                    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item() * group_size)  # log un-scaled loss
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    epoch_overflow_count = (
        amp_overflow_state["total_overflows"] - epoch_overflows_before
    )
    epoch_step_attempts = (
        amp_overflow_state["optimizer_step_attempts"] - epoch_attempts_before
    )
    epoch_steps_applied = (
        amp_overflow_state["optimizer_steps_applied"] - epoch_applied_before
    )
    stats.update({
        "amp_overflow_count": epoch_overflow_count,
        "amp_overflow_total": amp_overflow_state["total_overflows"],
        "amp_consecutive_overflows": amp_overflow_state["consecutive_overflows"],
        "optimizer_step_attempts": epoch_step_attempts,
        "optimizer_steps_applied": epoch_steps_applied,
        "optimizer_step_attempts_total": amp_overflow_state["optimizer_step_attempts"],
        "optimizer_steps_applied_total": amp_overflow_state["optimizer_steps_applied"],
        "amp_scale": (
            float(fp16_scaler.get_scale())
            if fp16_scaler is not None
            else None
        ),
    })
    return stats


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
