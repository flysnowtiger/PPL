"""Training and evaluation entry point for the original PPL implementation.

The training data flow and optimization behavior are preserved. Evaluation uses
the concatenated descriptor only, while runtime profiling remains observational.
"""

from __future__ import absolute_import, print_function

import argparse
import datetime
import os
import os.path as osp
import random
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

import data_manager
import models
import transforms as T
from config import cfg
from eval_metrics import evaluate_reranking
from losses import CrossEntropyLabelSmooth, TripletLoss
from lr_schedulers import WarmupMultiStepLR
from performance import (
    InferenceProfiler,
    format_elapsed,
    peak_gpu_memory_mb,
    print_inference_metrics,
    print_model_flops,
    reset_peak_gpu_memory,
    synchronize,
)
from samplers import RandomIdentitySampler
from utils import AverageMeter, DeepSupervision, Logger, make_optimizer
from vis_utils.visualization import visualization_sequence

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


def build_parser():
    """Parse the configuration source and optional YACS overrides only."""
    parser = argparse.ArgumentParser(description="PPL VideoReID")
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="./configs/resnet50.yml",
        help="path to the YAML configuration file",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="configuration overrides, for example DATASETS.SEQ_LEN 8",
    )
    return parser


def load_config(command_args):
    """Load one isolated configuration for the current run."""
    config = cfg.clone()
    if command_args.config_file:
        config.merge_from_file(command_args.config_file)
    config.merge_from_list(command_args.opts)
    config.freeze()
    return config


def build_runtime_options(config):
    """Map YAML values to the names used by the original implementation."""
    dataset = config.DATASETS.NAME.lower()
    return SimpleNamespace(
        gpu_device=",".join(str(gpu_id) for gpu_id in config.RUNTIME.GPU_IDS),
        log_dir=config.OUTPUT_DIR,
        data_dir=config.DATASETS.ROOT_DIR,
        model_dir=config.MODEL.PRETRAIN_DIR,
        dataset=dataset,
        split_id=config.DATASETS.SPLIT_ID,
        seq_len=config.DATASETS.SEQ_LEN,
        train_sampler=config.DATASETS.TRAIN_SAMPLE_METHOD,
        test_sampler=config.DATASETS.TEST_SAMPLE_METHOD,
        arch=config.MODEL.ARCH,
        model_mode=tuple(config.MODEL.MODES),
        str_mode=config.MODEL.STR_MODE,
        layer=config.MODEL.LAYER,
        part_num=config.MODEL.PART_NUM,
        lambda_1=config.MODEL.ATTENTION_LOSS_WEIGHT,
        lambda_2=config.MODEL.SPATIAL_TEMPORAL_LOSS_WEIGHT,
        visual=config.MODEL.VISUAL,
        batch_size=config.SOLVER.SEQS_PER_BATCH,
        triplet_distance=config.SOLVER.TRIPLET_DISTANCE,
        test_distance=config.TEST.DISTANCE,
        only_test=config.TEST.ONLY_TEST,
        test_path=config.TEST.WEIGHT,
        method_name=config.EXPERIMENT.METHOD_NAME,
        changed_thing=config.EXPERIMENT.CHANGED_THING,
    )


def should_evaluate(epoch, config):
    """Preserve the original epoch 1, 10, periodic, and final evaluations."""
    epoch_number = epoch + 1
    periodic_or_final = config.SOLVER.EVAL_PERIOD > 0 and (
        epoch_number % config.SOLVER.EVAL_PERIOD == 0
        or epoch_number == config.SOLVER.MAX_EPOCHS
    )
    return periodic_or_final or epoch in (0, 9)


def run_experiment(args_, cfg):
    """Build and run one filesystem-backed training or evaluation job."""
    args_.log_dir = os.path.join(args_.log_dir, args_.dataset)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    args_.log_dir = os.path.join(args_.log_dir, args_.method_name)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = args_.changed_thing + run_id
    args_.log_dir = os.path.join(args_.log_dir, run_id)
    if not os.path.exists(args_.log_dir):
        os.mkdir(args_.log_dir)
    print(args_.log_dir)
    if args_.only_test:
        test_save_dir = os.path.dirname(args_.test_path)
        sys.stdout = Logger(osp.join(test_save_dir, "log_test.txt"))
    else:
        sys.stdout = Logger(osp.join(args_.log_dir, "log_train.txt"))

    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    print(
        "=========================\nConfigs:{}\n=========================".format(cfg)
    )
    s = str(args_).split(", ")
    print("Fine-tuning detail:")
    for i in range(len(s)):
        print(s[i])
    print("=========================")

    os.environ["CUDA_VISIBLE_DEVICES"] = args_.gpu_device
    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    if use_gpu:
        print("Currently using GPU {}".format(args_.gpu_device))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args_.dataset))
    dataset = data_manager.init_dataset(
        root=args_.data_dir, name=args_.dataset, split_id=args_.split_id
    )
    dataset_num_train_pids = dataset.num_train_pids
    dataset_train = dataset.train
    dataset_query = dataset.query
    dataset_gallery = dataset.gallery
    from video_loader import VideoDataset

    if args_.visual:
        transform_train = T.Compose(
            [
                T.resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.to_tensor(),
                T.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform_train = T.Compose(
            [
                T.resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.to_tensor(),
                T.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.random_erasing(
                    probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN
                ),
            ]
        )

    transform_test = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pin_memory = True if use_gpu else False
    video_sampler = RandomIdentitySampler(
        dataset_train, num_instances=cfg.DATALOADER.NUM_INSTANCE
    )

    trainloader = DataLoader(
        VideoDataset(
            dataset_train,
            seq_len=args_.seq_len,
            sample=args_.train_sampler,
            transform=transform_train,
            dataset_name=args_.dataset,
        ),
        sampler=video_sampler,
        batch_size=args_.batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    print("Build dense sampler")
    queryloader_dense = DataLoader(
        VideoDataset(
            dataset_query,
            seq_len=args_.seq_len,
            sample="dense",
            transform=transform_test,
            max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM,
            dataset_name=cfg.DATASETS.NAME,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    galleryloader_dense = DataLoader(
        VideoDataset(
            dataset_gallery,
            seq_len=args_.seq_len,
            sample="dense",
            transform=transform_test,
            max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM,
            dataset_name=cfg.DATASETS.NAME,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    queryloader = DataLoader(
        VideoDataset(
            dataset_query,
            seq_len=args_.seq_len,
            sample="Begin_interval",
            transform=transform_test,
            max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM,
            dataset_name=cfg.DATASETS.NAME,
        ),
        batch_size=cfg.TEST.SEQS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(
            dataset_gallery,
            seq_len=args_.seq_len,
            sample="Begin_interval",
            transform=transform_test,
            max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM,
            dataset_name=cfg.DATASETS.NAME,
        ),
        batch_size=cfg.TEST.SEQS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print("Initializing model: {}".format(args_.arch))

    model = models.init_model(
        name=args_.arch,
        num_classes=dataset_num_train_pids,
        pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
        model_name=cfg.MODEL.NAME,
        seq_len=args_.seq_len,
        pretrained_model_dir=args_.model_dir,
        model_mode=args_.model_mode,
        str_mode=args_.str_mode,
        part_num=args_.part_num,
        layer=args_.layer,
        lambda_1=args_.lambda_1,
        lambda_2=args_.lambda_2,
        visual=args_.visual,
    )

    print_model_flops(model, args_.seq_len, cfg.INPUT.SIZE_TEST)
    print(
        "Model size: {:.5f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )

    model = nn.DataParallel(model)
    model.cuda()

    if args_.only_test:
        print("Loading checkpoint from '{}'".format(args_.test_path))
        print("load model... ")
        checkpoint = torch.load(args_.test_path)
        model.load_state_dict(checkpoint)
        print("this method: {:}".format(args_.method_name))

        print("==> Begin-interval Test")
        test(
            model,
            queryloader,
            galleryloader,
            use_gpu,
            args_.dataset,
            args_.test_distance,
            cfg.TEST.INFERENCE_WARMUP_BATCHES,
        )

        print("==> Dense Test")
        test(
            model,
            queryloader_dense,
            galleryloader_dense,
            use_gpu,
            args_.dataset,
            args_.test_distance,
            cfg.TEST.INFERENCE_WARMUP_BATCHES,
        )
    else:
        start_time = time.time()
        train_time = 0.0
        training_peak_memory = None
        reset_peak_gpu_memory(use_gpu)
        xent = CrossEntropyLabelSmooth(num_classes=dataset_num_train_pids)
        tent = TripletLoss(cfg.SOLVER.MARGIN, distance=args_.triplet_distance)

        optimizer = make_optimizer(cfg, model, mode=args_.model_mode)
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_ITERS,
            cfg.SOLVER.WARMUP_METHOD,
        )
        start_epoch = 0
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
            synchronize(use_gpu)
            epoch_train_start = time.perf_counter()
            if args_.visual:
                print("Loading checkpoint from '{}'".format(args_.test_path))
                print("load model... ")
                checkpoint = torch.load(args_.test_path)
                model.load_state_dict(checkpoint)
                train(
                    model,
                    trainloader,
                    xent,
                    tent,
                    optimizer,
                    use_gpu,
                    args_.visual,
                    args_.part_num,
                )

            print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
            print("current lr:", scheduler.get_lr()[0])

            train(
                model,
                trainloader,
                xent,
                tent,
                optimizer,
                use_gpu,
                args_.visual,
                args_.part_num,
            )
            synchronize(use_gpu)
            epoch_train_time = time.perf_counter() - epoch_train_start
            train_time += epoch_train_time
            current_peak_memory = peak_gpu_memory_mb(use_gpu)
            if current_peak_memory is not None:
                training_peak_memory = max(
                    training_peak_memory or 0.0, current_peak_memory
                )
            print(
                "Train time: epoch {:.3f}s; cumulative {:.3f}s ({})".format(
                    epoch_train_time, train_time, format_elapsed(train_time)
                )
            )
            scheduler.step()
            torch.cuda.empty_cache()

            if should_evaluate(epoch, cfg):
                print("==> Test")
                print("this method: {:}".format(args_.method_name))
                print("changed thing: {:}".format(args_.changed_thing))

                _, metrics = test(
                    model,
                    queryloader,
                    galleryloader,
                    use_gpu,
                    args_.dataset,
                    args_.test_distance,
                    cfg.TEST.INFERENCE_WARMUP_BATCHES,
                )
                rank1 = metrics[0]
                if epoch > 220:
                    state_dict = model.state_dict()
                    torch.save(
                        state_dict,
                        osp.join(
                            args_.log_dir,
                            "rank1_"
                            + str(rank1)
                            + "_checkpoint_ep"
                            + str(epoch + 1)
                            + ".pth",
                        ),
                    )
                    if (epoch + 1) == cfg.SOLVER.MAX_EPOCHS:
                        print("==> Dense Test")
                        print("This method is: {}".format(args_.method_name))
                        _, metrics = test(
                            model,
                            queryloader_dense,
                            galleryloader_dense,
                            use_gpu,
                            args_.dataset,
                            args_.test_distance,
                            cfg.TEST.INFERENCE_WARMUP_BATCHES,
                        )

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(
            "Training time: {:.3f}s ({})".format(train_time, format_elapsed(train_time))
        )
        print(
            "Peak GPU memory during training: {:.2f} MiB".format(
                training_peak_memory or 0.0
            )
        )
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, trainloader, xent, tent, optimizer, use_gpu, visual, part_num):
    """Run one training epoch with the original loss and update sequence."""
    model.train()
    xent_losses = AverageMeter()
    tent_losses = AverageMeter()
    regular_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        if use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if visual:
            atte_mask, simi_map = model(imgs)
            visualization_sequence(
                imgs, atte_mask, simi_map, batch_idx, part_num=part_num
            )
            if batch_idx >= 3:
                print(batch_idx)
        else:
            outputs, features, regular_loss = model(imgs)  #

            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(xent, outputs, pids, mode="CE-video")
            else:
                xent_loss = xent(outputs, pids)

            if isinstance(features, (tuple, list)):
                tent_loss = DeepSupervision(tent, features, pids, mode="Trip")
            else:
                tent_loss = tent(features, pids)

            xent_losses.update(xent_loss.item(), 1)
            tent_losses.update(tent_loss.item(), 1)
            regular_losses.update(regular_loss.mean().item(), 1)

            loss = xent_loss + tent_loss + regular_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), 1)

    print(
        "Batch {}/{}\t Loss {:.6f} ({:.6f}) ce Loss {:.6f} ({:.6f}), triplet Loss {:.6f} ({:.6f}), "
        "regular Loss {:.6f} ({:.6f})".format(
            batch_idx + 1,
            len(trainloader),
            losses.val,
            losses.avg,
            xent_losses.val,
            xent_losses.avg,
            tent_losses.val,
            tent_losses.avg,
            regular_losses.val,
            regular_losses.avg,
        )
    )
    return losses.avg


def extract_concatenated_features(model, loader, use_gpu, profiler):
    """Extract one concatenated descriptor per tracklet."""
    feature_batches = []
    person_ids, camera_ids, image_paths = [], [], []
    profiler.start_extraction()

    for images, pids, camids, paths in loader:
        image_paths.append(paths[0])
        video_count = images.size(0)
        if use_gpu:
            images = images.cuda()
            pids = pids.cuda()
            camids = camids.cuda()

        is_dense = images.dim() == 6
        if is_dense:
            batch_size, clips, seq_len, channels, height, width = images.size()
            if batch_size != 1:
                raise ValueError("Dense evaluation requires batch size 1")
            images = images.view(batch_size * clips, seq_len, channels, height, width)

        outputs = profiler.measure(
            lambda: model(images),
            video_count=video_count,
            sequence_count=images.size(0),
            frame_count=images.size(0) * images.size(1),
        )
        features = outputs[-1].data.cpu()
        features = features.view(-1, features.size(1))
        if is_dense:
            features = torch.mean(features, 0, keepdim=True)

        feature_batches.append(features)
        person_ids.extend(pids.data.cpu())
        camera_ids.extend(camids.data.cpu())
        torch.cuda.empty_cache()

    profiler.stop_extraction()
    return (
        torch.cat(feature_batches, 0),
        np.asarray(person_ids),
        np.asarray(camera_ids),
        image_paths,
    )


def test(
    model,
    queryloader,
    galleryloader,
    use_gpu,
    dataset_name,
    distance,
    warmup_batches,
    ranks=(1, 5, 10, 20),
):
    """Evaluate only the concatenated feature branch."""
    profiler = InferenceProfiler(use_gpu, warmup_batches=warmup_batches)
    profiler.start()

    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids, query_paths = extract_concatenated_features(
            model, queryloader, use_gpu, profiler
        )
        gf, g_pids, g_camids, gallery_paths = extract_concatenated_features(
            model, galleryloader, use_gpu, profiler
        )

    np.save("query_pathes", query_paths)
    np.save("gallery_pathes", gallery_paths)
    print(
        "Extracted concatenated query features, obtained {}-by-{} matrix".format(
            qf.size(0), qf.size(1)
        )
    )

    if dataset_name == "mars":
        # MARS gallery must include query tracklets for queries lacking ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    print(
        "Extracted concatenated gallery features, obtained {}-by-{} matrix".format(
            gf.size(0), gf.size(1)
        )
    )
    print("Computing distance matrix")
    be_cmc, metrics = evaluate_reranking(
        qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, distance
    )
    print_inference_metrics(profiler.summarize())
    return metrics, be_cmc


def main():
    """Load configuration and dispatch the selected experiment."""
    command_args = build_parser().parse_args()
    config = load_config(command_args)
    options = build_runtime_options(config)
    run_experiment(options, config)


if __name__ == "__main__":
    main()
