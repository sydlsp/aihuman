import argparse
import copy
import logging
import math
import os
import os.path as osp
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from tqdm.auto import tqdm
from einops import rearrange

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL,DDIMScheduler
from transformers import CLIPVisionModelWithProjection

from dataset_related.dataset_write_2 import MyDataset,collate_fn # 用dataset_write_1版本的
from AniPortrait.src.models.mutual_self_attention import ReferenceAttentionControl
from AniPortrait.src.models.unet_2d_condition import UNet2DConditionModel
from AniPortrait.src.models.unet_3d import UNet3DConditionModel
from AniPortrait.src.models.pose_guider import PoseGuider
from AniPortrait.src.utils.util import (
    delete_additional_ckpt,
    import_filename,
    read_frames,
    save_videos_grid,
    seed_everything,
)
"""
train2_1和train1_2是匹配的
"""


warnings.filterwarnings("ignore")

check_min_version("0.10.0.dev0")

logger=get_logger(__name__,log_level="INFO")

"""
  这里的net应该和第一阶段的net是一致的
"""
class Net(nn.Module):
    def __init__(self,
                 reference_unet:UNet2DConditionModel,
                 denoising_unet:UNet3DConditionModel,
                 pose_guider:PoseGuider,
                 reference_control_writer,
                 reference_control_reader,):
        super().__init__()
        self.reference_unet=reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(self,
                noisy_latents,
                time_steps,
                ref_image_latents,
                clip_image_embeds,
                # pose_image,
                # ref_pose_img, 这里是本来要给pose_guider的参数现在我们变复杂了，不用这个
                ref_image_mesh,
                ref_image_keypoint,
                pose_image_mesh,
                pose_image_keypoint,  # 注意一下这里我们和源代码在pose和ref的位置是不一样的
                uncond_fwd:bool=False,
                device="cuda",):
        """"""
        ref_mesh_tensor=ref_image_mesh #.to(device=device)
        ref_keypoint_tensor=ref_image_keypoint #.to(device=device)
        pose_mesh_tensor=pose_image_mesh #.to(device=device)
        pose_keypoint_tensor=pose_image_keypoint #.to(device=device)
        # print("__________________________________________________________________")
        # print(ref_mesh_tensor.shape)
        # print(ref_keypoint_tensor.shape)
        # print(pose_mesh_tensor.shape)
        # print(pose_keypoint_tensor.shape)
        # torch.Size([1, 3, 256, 256])
        # torch.Size([1, 3, 256, 256])
        # torch.Size([1, 3, 1, 256, 256])
        # torch.Size([1, 3, 1, 256, 256])

        # pose_guider前面放的是pose 后
        mesh_fea=self.pose_guider(pose_mesh_tensor,ref_mesh_tensor)
        keypoint_fea=self.pose_guider(pose_keypoint_tensor,ref_keypoint_tensor)

        pose_fea=mesh_fea+keypoint_fea

        if not uncond_fwd:
            ref_timesteps=torch.zeros_like(time_steps)
            self.reference_unet(ref_image_latents,ref_timesteps,encoder_hidden_states=clip_image_embeds,return_dict=False)
            self.reference_control_reader.update(self.reference_control_writer)
        else:
            pass

        model_pred=self.denoising_unet(noisy_latents,time_steps,pose_cond_fea=pose_fea,encoder_hidden_states=clip_image_embeds).sample

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

def main(cfg):
    kwargs=DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator=Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 在所有进程中记录accelerator的状态
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name=cfg.exp_name

    # 实验名称和输出目录拼接成保存目录
    save_dir = f"{cfg.output_dir}/{exp_name}"

    # 要检查当前进程是不是主进程，只有主进程才会创建保存目录
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    sample_dir = os.path.join(save_dir, 'samples')

    if accelerator.is_main_process and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 这里这个名字起的和推理没什么关系，是在加载Unet中要用到的一些参数
    inference_config_path = "/remote-home/yfsong/shipu/mycode/AniPortrait/configs/inference/inference_v2.yaml"
    infer_config = OmegaConf.load(inference_config_path)

    if cfg.weight_dtype=="fp16":
        weight_dtype=torch.float16
    elif cfg.weight_dtype=="fp32":
        weight_dtype=torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs=OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )


    # 用于验证的噪声调度器，这里应该也是用不到的
    # val_noise_scheduler=DDIMScheduler(**sched_kwargs)

    # 用于训练的噪声调度器
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    # 图像编码器
    image_enc=CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dtype=weight_dtype, device="cuda")

    # vae自动编码器
    vae=AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda", dtype=weight_dtype)

    reference_unet=UNet2DConditionModel.from_pretrained(cfg.base_model_path,subfolder="unet",).to(device="cuda", dtype=weight_dtype)

    denoising_unet=UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs),
    ).to(device="cuda")

    # poseguider 这里要注意一下我们还是沿用Aniprotrait的
    pose_guider=PoseGuider(noise_latent_channels=320).to(device="cuda",dtype=weight_dtype)

    stage1_ckpt_dir=cfg.stage1_ckpt_dir
    stage1_ckpt_step = cfg.stage1_ckpt_step

    # 加载权重，这部分是加载ckpt1训练完的权重

    denoising_unet.load_state_dict(
        torch.load(os.path.join(stage1_ckpt_dir,f"denoising_unet-{stage1_ckpt_step}.pth"),map_location="cpu"),
        strict=False
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

    # 把模型权重全部冻起来
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    pose_guider.requires_grad_(False)

    # 在第二阶段的训练中，只有motion module模块是可学习的
    for name,module in denoising_unet.named_modules():
        if "motion_module" in name:
            for params in module.parameters():
                params.requires_grad=True

    reference_control_writer=ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full"
    )

    reference_control_reader=ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net=Net(
        reference_unet,
        denoising_unet,
        pose_guider,
        reference_control_writer,
        reference_control_reader,
    )

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
    trainable_params=list(filter(lambda p:p.requires_grad,net.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")

    # 实例化优化器
    optimizer=optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # 学习率调度器
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # 读数据，这里就是把之前is_image=True改成False
    train_dataset=MyDataset(**cfg.data,is_image=False)

    train_dataloader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_bs,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # 用accelerator再包起来
    (net,optimizer,train_dataloader,lr_scheduler)=accelerator.prepare(net,optimizer,train_dataloader,lr_scheduler)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )

    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
        )

    """
    开始训练
    """
    total_batch_size = (
            cfg.train_bs
            * accelerator.num_processes
            * cfg.solver.gradient_accumulation_steps
    )

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

    global_step = 0
    first_epoch = 0

    # 从检查点恢复模型
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

    # 进度条显示

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch,num_train_epochs):
        train_loss=0.0
        t_data_start=time.time()
        for step,batch in enumerate(train_dataloader):
            t_data=time.time()-t_data_start

            with accelerator.accumulate(net):

                pixel_values_vid=batch["pixel_values"].to(weight_dtype)
                with torch.no_grad():
                    video_length=pixel_values_vid.shape[1]

                    pixel_values_vid=rearrange(pixel_values_vid,"b f c h w ->(b f) c h w")

                    latents=vae.encode(pixel_values_vid).latent_dist.sample()

                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                    # 这是大家都在做的一个操作，就这样吧，可能效果会好一点吧
                    latents = latents * 0.18215

                noise=torch.randn_like(latents)

                if cfg.noise_offset > 0:
                     noise+=cfg.noise_offset*torch.randn(
                         (latents.shape[0], latents.shape[1], 1, 1, 1),
                         device=latents.device,
                     )

                bsz=latents.shape[0]

                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )

                timesteps=timesteps.long()

                """
                  这里我们变量的写法和train1_2中的变量名保持一致
                """
                pose_img_mesh=batch["pixel_values_mesh"] # [bs,f,c,H,W]
                pose_img_mesh=pose_img_mesh.transpose(1,2) # [bs,c,f,H,W]

                pose_img_keypoint=batch["pixel_values_keypoint"]
                pose_img_keypoint=pose_img_keypoint.transpose(1,2)

                ref_img_mesh=batch["pixel_values_ref_mesh"]
                ref_img_keypoint=batch["pixel_values_ref_keypoint"]

                uncond_fwd=random.random()<cfg.uncond_ratio
                clip_image_list=[]
                ref_image_list=[]

                for batch_idx,(ref_img,clip_img) in enumerate(
                    zip(
                        batch["pixel_values_ref_img"],
                        batch["clip_ref_image"]
                    )
                ):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_image_list.append(ref_img)

                with torch.no_grad():
                    ref_img = torch.stack(ref_image_list, dim=0).to(dtype=vae.dtype, device=vae.device)

                    # 参考图过vae
                    ref_image_latents=vae.encode(ref_img).latent_dist.sample()

                    ref_image_latents=ref_image_latents*0.18215

                    # ref图像的clip编码
                    clip_img=torch.stack(clip_image_list,dim=0).to(dtype=image_enc.dtype,device=image_enc.device)
                    clip_img=clip_img.to(device="cuda",dtype=weight_dtype)
                    clip_image_embeds=image_enc(clip_img.to("cuda",dtype=weight_dtype)).image_embeds # 这里做了两次到cuda上的操作
                    clip_image_embeds=clip_image_embeds.unsqueeze(1) # [bs,1,d]

                noisy_latents=train_noise_scheduler.add_noise(latents,noise,timesteps)

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

                model_pred=net(noisy_latents,timesteps,ref_image_latents,clip_image_embeds,ref_img_mesh,ref_img_keypoint,pose_img_mesh,pose_img_keypoint,uncond_fwd=uncond_fwd) # 这里先这样写，在这里我们补上了uncond_fwd参数

                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr=compute_snr(train_noise_scheduler,timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]/ snr
                    )

                    loss=F.mse_loss(model_pred.float(),target.float(),reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape))))* mse_loss_weights
                    )

                    loss=loss.mean()

                avg_loss = accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params,cfg.solver.max_grad_norm,)
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

                # 同样的关于评估的我们不写
            logs={"step_loss":loss.detach().item(),
                  "lr": lr_scheduler.get_last_lr()[0],
                  "td": f"{t_data:.2f}s",
                  }

            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        if accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
            delete_additional_ckpt(save_dir, 1)
            accelerator.save_state(save_path)
            # 这里save_checkpoint只保存motion_module
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "motion_module",
                global_step,
                total_limit=3,
            )

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

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="/remote-home/yfsong/shipu/mycode/train_config/train_2.yaml")
    args=parser.parse_args()

    if args.config[-5:]==".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)

# accelerate launch --config_file /remote-home/yfsong/shipu/mycode/one_gpu_default_config.yaml train_2_1.py








