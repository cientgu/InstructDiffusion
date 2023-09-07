# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Removed Pytorch-lightning and supported deepspeed by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import argparse, os, sys, datetime, glob
import numpy as np
import time
import json
import pickle
import wandb
import deepspeed

from packaging import version
from omegaconf import OmegaConf
from functools import partial
from PIL import Image

from timm.utils import AverageMeter

import torch
import torchvision
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
sys.path.append("./stable_diffusion")

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from utils.logger import create_logger
from utils.utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper
from utils.deepspeed import create_ds_config


def wandb_log(*args, **kwargs):
    if dist.get_rank() == 0:
        wandb.log(*args, **kwargs)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--amd",
        action="store_true",
        default=False,
        help="amd",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        # required=False,
        default=int(os.environ.get('LOCAL_RANK', 0)),
        help="local rank for DistributedDataParallel",
    )
    return parser


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig():
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            if "target" in train:
                self.dataset_configs["train"] = train
                self.train_dataloader = self._train_dataloader
            else:
                for ds in train:
                    ds_name = str([key for key in ds.keys()][0])
                    self.dataset_configs[ds_name] = ds
                self.train_dataloader = self._train_concat_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_concat_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['ds1'], Txt2ImgIterableBaseDataset)

        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        concat_dataset = []
        for ds in self.datasets.keys():
            concat_dataset.append(self.datasets[ds])

        concat_dataset = ConcatDataset(concat_dataset)
        sampler_train = torch.utils.data.DistributedSampler(
            concat_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(concat_dataset, batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        sampler_train = torch.utils.data.DistributedSampler(
            self.datasets["train"], num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, sampler=sampler_train,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)


def train_one_epoch(config, model, model_ema, data_loader, val_data_loader, optimizer, epoch, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    accumul_steps = config.trainer.accumulate_grad_batches
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()
    loss_scale_meter_min = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, batch in enumerate(data_loader):
        batch_size = batch['edited'].shape[0]
        
        if config.model.params.deepspeed != '':
            loss, _ = model(batch, idx, accumul_steps)
            model.backward(loss)

            model.step()
            loss_scale = optimizer.cur_scale
            grad_norm = model.get_global_grad_norm()

            with torch.no_grad():
                if idx % config.trainer.accumulate_grad_batches == 0:
                    model_ema(model)

            loss_number = loss.item()
        else:
            with amp.autocast(enabled=config.model.params.fp16):
                loss, _ = model(batch, idx, accumul_steps)

            if config.trainer.accumulate_grad_batches > 1:
                loss = loss / config.trainer.accumulate_grad_batches
                scaler.scale(loss).backward()
                # loss.backward()
                if config.trainer.clip_grad > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.trainer.accumulate_grad_batches == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    # scaler.unscale_grads()
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if config.trainer.clip_grad > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                # lr_scheduler.step_update(epoch * num_steps + idx)
            
            loss_scale = scaler.get_scale()
            loss_number = loss.item() * config.trainer.accumulate_grad_batches

        torch.cuda.synchronize()
        
        loss_meter.update(loss_number, batch_size)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        else:
            norm_meter.update(0.0)

        loss_scale_meter.update(loss_scale)
        # loss_scale_meter.update(0.0)
        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

        if (epoch * num_steps + idx) % 100 == 0:
            log_message = dict(
                lr=optimizer.param_groups[0]['lr'], 
                time=batch_time.val, 
                epoch=epoch, 
                iter=idx, 
                loss=loss_meter.val, 
                grad_norm=norm_meter.val, 
                loss_scale=loss_scale_meter.val, 
                memory=torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                global_iter=epoch * num_steps + idx)

            # log_message.update({'ref_img': wandb.Image(unnormalize(img[:8].cpu().float())), 'mask': wandb.Image(mask[:8].cpu().float().unsqueeze(1))})
            # if x_rec is not None:
                # log_message.update({'rec_img': wandb.Image(unnormalize(x_rec[:8].cpu().float()))})
            wandb_log(
                data=log_message,
                step=epoch * num_steps + idx,
            )

        if idx == num_steps - 1:
            with torch.no_grad():
                model_ema.store(model.parameters())
                model_ema.copy_to(model)
                for val_idx, batch in enumerate(val_data_loader):
                    batch_size = batch['edited'].shape[0]

                    loss, _ = model(batch, -1, 1)

                    loss_number = loss.item()
                    val_loss_meter.update(loss_number, batch_size)
                    if val_idx % 10 == 0:
                        logger.info(
                            f'Val: [{val_idx}/{len(val_data_loader)}]\t'
                            f'loss {val_loss_meter.val:.4f} ({val_loss_meter.avg:.4f})\t')
                    if val_idx == 50:
                        break
                model_ema.restore(model.parameters())

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    assert opt.name
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    nowname = f"{cfg_name}_{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if opt.amd:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)
    else:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    
    seed = opt.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    # init and save configs
    # config: the configs in the config file
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if config.model.params.deepspeed != '':
        create_ds_config(opt, config, cfgdir)

    if dist.get_rank() == 0:
        run = wandb.init(
            id=nowname,
            name=nowname,
            project='readoutpose',
            config=OmegaConf.to_container(config, resolve=True),
        )

    logger = create_logger(output_dir=logdir, dist_rank=dist.get_rank(), name=f"{nowname}")
    
    resume_file = auto_resume_helper(config, ckptdir)
    if resume_file:
        resume = True
        logger.info(f'resume checkpoint in {resume_file}')
    else:
        resume = False
        logger.info(f'no checkpoint found in {ckptdir}, ignoring auto resume')

    # model
    model = instantiate_from_config(config.model)
    model_ema = LitEma(model, decay_resume=config.model.params.get('ema_resume', 0.9999))

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    data_loader_train = data.train_dataloader()
    data_loader_val = data.val_dataloader()

    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate    
    ngpu = dist.get_world_size()
    if 'accumulate_grad_batches' in config.trainer:
        accumulate_grad_batches = config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    if not opt.amd:
        model.cuda()

    if config.model.params.fp16 and config.model.params.deepspeed == '':
        scaler = amp.GradScaler()
        param_groups = model.parameters()
    else:
        scaler = None
        param_groups = model.parameters()

    if config.model.params.deepspeed != '':
        model, optimizer, _, _ = deepspeed.initialize(
            args=config,
            model=model,
            model_parameters=param_groups,
            dist_init_required=False,
        )
        for name, param in model.named_parameters():
            param.global_name = name
        model_without_ddp = model
        lr_scheduler = None
        model_ema = model_ema.to(next(model.parameters()).device)
    else:
        optimizer, lr_scheduler = model.configure_optimizers()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], broadcast_buffers=False)
        model_without_ddp = model.module
    # print(optimizer.param_groups[1])
    if opt.resume != '':
        resume_file = opt.resume
    if resume_file:
        _, start_epoch = load_checkpoint(resume_file, config, model_without_ddp, model_ema, optimizer, lr_scheduler, scaler, logger)
    else:
        start_epoch = 0

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, config.trainer.max_epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, model, model_ema, data_loader_train, data_loader_val, optimizer, epoch, lr_scheduler, scaler)
        if epoch % config.trainer.save_freq == 0:
            save_checkpoint(ckptdir, config, epoch, model_without_ddp, model_ema, 0., optimizer, lr_scheduler, scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
