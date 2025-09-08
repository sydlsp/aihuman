import argparse
import copy
import logging
import math
import os
import pdb
import os.path as osp
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from src.dataset.dataset_face import FaceDataset, FaceDatasetValid, collate_fn
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        ref_pose_img,
        uncond_fwd: bool = False,
    ):
        """
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pixel_values_pose,
        pixel_values_ref_pose,
        """

        # pixel_values_pose
        pose_cond_tensor = pose_img.to(device="cuda")
        # pixel_values_ref_pose
        ref_pose_tensor = ref_pose_img.to(device="cuda")
        # 对应论文中Reference Pose Image和Target Pose Images一起被放入PoseGuider中
        pose_fea = self.pose_guider(pose_cond_tensor, ref_pose_tensor)

        # 如果按概率是无条件的前向传播
        # 那么ref_timesteps就全是0
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            # Unet2DConditionalModel 输入是pose_feature
            # 条件是clip_image_embeds [batch_size,1,d] d可能是768
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            # 这里要好好研究研究在，这里对应的是reference_net的denosing_net的
            self.reference_control_reader.update(self.reference_control_writer)
        else:
            pass
        
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    vae,
    image_enc,
    net,
    scheduler,
    accelerator,
    width,
    height,
    clip_length=24,
    generator=None,
    valid_dataset=None
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    pose_guider = ori_net.pose_guider

    if generator is None:
        generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)
    tmp_denoising_unet = tmp_denoising_unet.to(dtype=torch.float16)

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    dataset_len = len(valid_dataset)
    sample_idx = [random.randint(0, dataset_len) for _ in range(2)]

    results = []
    for idx in sample_idx:
        sample = valid_dataset[idx]

        # 只能接受一个numpy数组，给他列表是要用for循环的

        ref_image_pil = Image.fromarray(sample['ref_img']).convert("RGB")
        pose_images = [Image.fromarray(sample['pixel_values_pose'][idx]).convert("RGB") for idx in range(sample['pixel_values_pose'].shape[0])]
        gt_images = [Image.fromarray(sample['tar_gt'][idx]).convert("RGB") for idx in range(sample['tar_gt'].shape[0])]

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        
        pose_tensor_list = []
        ref_tensor_list = []
        gt_tensor_list = []
        pose_list = []

        for pose_image_pil in pose_images[:clip_length]:
            pose_tensor_list.append(pose_transform(pose_image_pil))
            ref_tensor_list.append(pose_transform(ref_image_pil))
        for gt_image_pil in gt_images[:clip_length]:
            gt_tensor_list.append(pose_transform(gt_image_pil))

        pose_list = sample['pixel_values_pose'][:clip_length]
        ref_pose = sample['pixel_values_ref_pose']

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1) # (c, f, h, w)

        ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
        ref_tensor = ref_tensor.transpose(0, 1) # (c, f, h, w)
        
        gt_tensor = torch.stack(gt_tensor_list, dim=0)  # (f, c, h, w)
        gt_tensor = gt_tensor.transpose(0, 1) # (c, f, h, w)

        pipeline_output = pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            width,
            height,
            clip_length,
            25,
            3.5,
            generator=generator,
        )
        video = pipeline_output.videos

        # Concat it with pose tensor
        pose_tensor = pose_tensor.unsqueeze(0)
        ref_tensor = ref_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
        video = torch.cat([ref_tensor, pose_tensor, video, gt_tensor], dim=0)

        results.append({"name": f"sample_{idx}", "vid": video})

    del tmp_denoising_unet
    del pipe
    torch.cuda.empty_cache()

    return results


def main(cfg):
    # 创建类的实例，DistributedDataParallelKwargs 用来存储DistributedDataParallel的关键词参数
    # DistributedDataParallel用来实现在多个GPU上处理并行数据
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    # 创建Accelerator的实例，用来处理分布式训练
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps, # 指定梯度累积的步数，可以减少显存的使用
        mixed_precision=cfg.solver.mixed_precision, # 指定是否使用混合精度训练，这里是'fp16'
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.

    # 用logging模块来配置基本的日志设置
    # format定义了日志的输出格式，分别记录日志的时间、级别、名称和消息内容
    # datefmt定义了时间(asctime)的格式
    # level设置了日志的级别为INFO，这里只有INFO以上的日志会被记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 在所有进程中记录accelerator的状态信息
    logger.info(accelerator.state, main_process_only=False)

    # 设置transformer和diffusers库的日志级别
    # 如果当前进程是本地主进程，transformer的日志级别被设置为警告，diffusers的日志级别被设置为信息
    # 如果当前进程不是本地主进程，transformer和diffusers的日志级别都被设置为错误
    # 设置日志级别意味着只有大于等于该日志级别的日志才会被记录
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # 设置随机种子
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # 读取实验名称
    exp_name = cfg.exp_name

    # 实验名称和输出目录拼接成保存目录
    save_dir = f"{cfg.output_dir}/{exp_name}"

    # 还是要检查当前进程是不是主进程，只有是主进程了才会创建保存目录
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 创建采样目录
    sample_dir = os.path.join(save_dir, 'samples')

    # 同样的，只有在主进程中才能创建采样目录
    if accelerator.is_main_process and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)


    inference_config_path = "./configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
    # 将一个配置对象转化为python字典
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    # 如果是启用零信噪比模式，那么对噪声调度器的配置参数进行更新
    # .update()其实就是在原有的字典后面补充新的键值对
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )

    # 用于验证的噪声调度器
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # 用于训练的噪声调度器，其实都是DDIMScheduler
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # 图像编码器
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")

    # vae自动编码器
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    # reference_unet是一个2D条件模型
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)

    # 去噪unet是一个3D条件模型，.from_pretrained_2d是从预训练的2d模型加载权重
    # 需要两个参数，分别执行基础模型和mm模型的路径，mm模型是motion_module
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs
        ),
    ).to(device="cuda")

    # pose_guider网络
    pose_guider = PoseGuider(noise_latent_channels=320).to(device="cuda", dtype=weight_dtype)

    stage1_ckpt_dir = cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step

    # 加载模型权重，strict=False意味着即使当前模型结构和保存的模型权重不完全匹配也不会报错
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )

    # Freeze
    # 把模型权重全部冻起来
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # Set motion module learnable
    # 仅仅设置motion module模块是可学习的
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True

    # 下面这两个类实例化的主要功能是修改reference_unet模型的自注意力和组归一化来实现
    # 对注意力的控制
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    # 模型汇总成一个总模型
    net = Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

    # 是否在reference_unet和denosing_unet中使用内存高效的注意力机制，但有个问题，这样写
    # Net已经初始化了，这个时候再去设置reference_unet和denoising_unet的内存高效注意力机制是否是有效的
    # 这里有效的根本原因是python是在引用传递，所以这里的reference_unet和denoising_unet和Net中的是同一个
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # 是否启用梯度检查点
    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    # 设置学习率
    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    # 设置优化器
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # 获取网络中所有可学习的参数
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")

    # 实例化优化器，这里只把可学习的参数放入优化器中
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    # 学习率调度器写法要学
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # 读数据
    train_dataset = FaceDataset(**cfg.data, is_image=False)
    valid_dataset = FaceDatasetValid(**cfg.data, is_image=False)

    # 创建dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.train_bs, 
        shuffle=True, 
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
    )


    # Prepare everything with our `accelerator`.
    # 用accelerator.prepare准备好要用到的东西
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # 计算总的训练步数
    # 计算每个epoch的更新步数，用训练数据加载器的大小/梯度累积步数的上限
    # 梯度累积是一种常用的训练技巧，可以在内存限制下训练更大的批次
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs

    # 训练的总轮数：最大训练步数/每个epoch的更新步数向上取整
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
        )

    # Train!
    # 开始训练
    # 总的batch_size=每张卡上的batch_size*加速器的进程数*梯度累积步数
    total_batch_size = (
        cfg.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    # 在控制台输出部分训练信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    # 这两个变量的作用其实是为了下面从检查点恢复时用的
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # 是否从预先的检查点恢复模型的训练状态
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    # 只在每台机器上显示一次进度条
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")


    # 训练开始
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(train_dataloader):
            t_data = time.time() - t_data_start

            # accelerator.accumulate用于梯度累积
            with accelerator.accumulate(net):
                # Convert videos to latent space
                # batch["pixel_values"] [batch_size,n_frames,channels,height,width]
                pixel_values_vid = batch["pixel_values"].to(weight_dtype)
                with torch.no_grad():
                    video_length = pixel_values_vid.shape[1]

                    # 修改输入视频的形状
                    pixel_values_vid = rearrange(
                        pixel_values_vid, "b f c h w -> (b f) c h w"
                    )

                    # 将视频映射到潜在空间中
                    latents = vae.encode(pixel_values_vid).latent_dist.sample()

                    # 再变一下形状，告诉了f这里就可以自动计算b的大小，然后做形状变换
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=video_length
                    )
                    # 这是大家都在做的一个操作，就这样吧，可能效果会好一点吧
                    latents = latents * 0.18215

                # 生成原始噪声
                noise = torch.randn_like(latents)
                # 根据配置文件在原有的噪声上加一些额外的噪声，调整一下噪声的强度
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device,
                    )
                bsz = latents.shape[0]

                # Sample a random timestep for each video
                # 为每个视频选一个随机时间步
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                # 把timestep数据类型修改为long
                timesteps = timesteps.long()

                # 姿势图，姿势图和pixel_values的形状是一样的
                pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)
                # 姿势图修改一下形状
                pixel_values_pose = pixel_values_pose.transpose(
                    1, 2
                )  # (bs, c, f, H, W)

                # pixel_values_ref_pose [batch_size,3,256,256]
                pixel_values_ref_pose = batch["pixel_values_ref_pose"]

                # 随机选择是否进行无条件前向传播
                uncond_fwd = random.random() < cfg.uncond_ratio
                clip_image_list = []
                ref_image_list = []

                for batch_idx, (ref_img, clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"], # [batch_size,3,256,256]
                        batch["clip_ref_image"], # 这里可以先理解为[batch_size,2,224,224],clip_ref_images是pixel_values_ref_img要放入clip的版本
                    )
                ):
                    # 无条件传播的话，就把clip_img设置为0
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    # 参考图也要过vae
                    ref_image_latents = vae.encode(
                        ref_img
                    ).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    # ref图像的clip编码
                    clip_img = torch.stack(clip_image_list, dim=0).to(
                        dtype=image_enc.dtype, device=image_enc.device
                    )
                    clip_img = clip_img.to(device="cuda", dtype=weight_dtype)

                    # 用clip对ref图像进行编码
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    # 这样的操作还是在对形状
                    clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                # add noise
                # 在latents上加噪
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                
                # Get the target for loss depending on the prediction type
                # 根据噪声调度器的类型来确定损失函数的目标
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    # 用于计算噪声图像的速度，预测噪声速度可以提升图像的生成质量
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                # ---- Forward!!! -----
                model_pred = net(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    clip_image_embeds,
                    pixel_values_pose,
                    pixel_values_ref_pose,
                    uncond_fwd=uncond_fwd,
                )

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (global_step % cfg.val.validation_steps == 0) or (global_step in cfg.val.validation_steps_tuple):
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            vae=vae,
                            image_enc=image_enc,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.sample_size[0],
                            height=cfg.data.sample_size[1],
                            clip_length=cfg.data.sample_n_frames,
                            generator=generator,
                            valid_dataset=valid_dataset
                        )

                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            vid = sample_dict["vid"]
                            out_file = os.path.join(sample_dir, f'{global_step:06d}-{sample_name}.gif')
                            save_videos_grid(vid, out_file, n_rows=4)

                        reference_control_writer = ReferenceAttentionControl(
                            reference_unet,
                            do_classifier_free_guidance=False,
                            mode="write",
                            fusion_blocks="full",
                        )
                        reference_control_reader = ReferenceAttentionControl(
                            denoising_unet,
                            do_classifier_free_guidance=False,
                            mode="read",
                            fusion_blocks="full",
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
        # save model after each epoch
        if accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # save motion module only
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "motion_module",
                global_step,
                total_limit=3,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, save_path)


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    # video = self.vae.decode(latents).sample
    video = []
    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx : frame_idx + 1]).sample)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float().numpy()
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
