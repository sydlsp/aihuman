import argparse
import logging
import pdb
import random

from einops import rearrange
import math
import os.path
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm.auto import tqdm

import transformers
import diffusers
from diffusers.utils import check_min_version
from diffusers import AutoencoderKL,DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from transformers import CLIPVisionModelWithProjection
from omegaconf import OmegaConf


from AniPortrait.src.utils.util import seed_everything
from AniPortrait.src.models.unet_2d_condition import UNet2DConditionModel
from AniPortrait.src.models.unet_3d import UNet3DConditionModel
from AniPortrait.src.models.mutual_self_attention import ReferenceAttentionControl
from AniPortrait.src.utils.util import delete_additional_ckpt
from redance.poseguider_4cond import get_poseguider
from dataset_related.dataset_write_7 import MyDataset,collate_fn# 这里别忘记改了
"""
train_dataloader在不同的训练阶段是有不同表现形式的，在第一阶段没有引入motion module的时候我们采用的其实是图片-图片的训练方式
"""

"""
配合dataset_write_7 使用"""

torch.set_printoptions(threshold=np.inf)
# 在这里设置单卡什么卡可见
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.filterwarnings("ignore")

check_min_version("0.10.0.dev0")

logger=get_logger(__name__,log_level="INFO")

"""
我们使用RealiseDance的条件融合策略
采用四个控制条件来控制手部动作，还是用RealiseDance的条件融合策略
"""


class Net(nn.Module):
    def __init__(self,
                 reference_unet: UNet2DConditionModel,
                 denoising_unet: UNet3DConditionModel,
                 pose_guider,
                 reference_control_writer,
                 reference_control_reader,
                 ):
        super().__init__()
        self.reference_unet=reference_unet
        self.denoising_unet=denoising_unet
        self.pose_guider=pose_guider
        self.reference_control_writer=reference_control_writer
        self.reference_control_reader=reference_control_reader

    def forward(self,
                noisy_latents,
                time_steps,
                ref_image_latents,
                clip_image_embeds,
                # ref_image_mesh,
                # ref_image_keypoint,
                # ref_image_hand,  # 这三个条件我们都不要了
                pose_image_mesh,
                pose_image_keypoint,  # 注意一下这里我们和源代码在pose和ref的位置是不一样的
                pose_image_hand,  # pose_hand
                pose_image_hdkp,  # 这里是新增的手部关键点
                uncond_fwd:bool=False,
                device="cuda",):
        """"""
        # ref_mesh_tensor=ref_image_mesh #.to(device=device)
        # ref_keypoint_tensor=ref_image_keypoint #.to(device=device)
        # ref_hand_tensor=ref_image_hand
        pose_mesh_tensor=pose_image_mesh #.to(device=device)
        pose_keypoint_tensor=pose_image_keypoint #.to(device=device)
        pose_hand_tensor=pose_image_hand
        pose_hdkp_tensor=pose_image_hdkp

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
        # mesh_fea=self.pose_guider(pose_mesh_tensor,ref_mesh_tensor)
        # keypoint_fea=self.pose_guider(pose_keypoint_tensor,ref_keypoint_tensor)
        # hand_fea=self.pose_guider(pose_hand_tensor,ref_hand_tensor)  # 这里是新增的手部mesh

        frame=pose_mesh_tensor.shape[2]
        pose_mesh_tensor=rearrange(pose_mesh_tensor,"b c f h w -> (b f) c h w")
        pose_keypoint_tensor=rearrange(pose_keypoint_tensor,"b c f h w -> (b f) c h w")
        pose_hand_tensor=rearrange(pose_hand_tensor,"b c f h w -> (b f) c h w")
        pose_hdkp_tensor=rearrange(pose_hdkp_tensor,"b c f h w -> (b f) c h w")

        pose_fea=self.pose_guider(pose_mesh_tensor,pose_keypoint_tensor,pose_hand_tensor,pose_hdkp_tensor)

        pose_fea=[rearrange(pe,"(b f) c h w -> b c f h w",f=frame) for pe in pose_fea]

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
    kwargs=DistributedDataParallelKwargs(find_unused_parameters=True)

    # 实例化accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        kwargs_handlers=[kwargs],
    )

    # 配置一下日志记录器的基本设置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # .info用来记录日志消息
    logger.info(accelerator.state, main_process_only=False)

    # 设置transformers和diffusers的日志级别
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.seed is not None:
        print("seed is not None")
        seed_everything(cfg.seed)

    exp_name=cfg.exp_name

    # 看一下这两个文件夹是装什么的
    save_dir = f"{cfg.output_dir}/{exp_name}" # 这个文件夹是到时候放checkpoint的
    if accelerator.is_local_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sample_dir = os.path.join(save_dir, 'samples')
    if accelerator.is_main_process and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # 选择权重数据类型
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
    # 将noise_scheduler相关配置转化为字典
    sched_kwargs=OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    print(sched_kwargs)


    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )

    # 这里初始化训练和验证时要用到的噪声调度器
    val_noise_scheduler=DDIMScheduler(**sched_kwargs)

    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler=DDIMScheduler(**sched_kwargs)
    train_noise_scheduler.set_timesteps(num_inference_steps=1,device=accelerator.device)
    train_noise_scheduler.alphas_cumprod=train_noise_scheduler.alphas_cumprod.to(device=accelerator.device)




    # 初始化VAE
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device=accelerator.device,dtype=weight_dtype)
    # 初始化reference_unet
    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path,subfolder="unet").to(device=accelerator.device)
    # 初始化denoising_unet
    denoising_unet=UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path,"",subfolder="unet",
                                                           unet_additional_kwargs={"use_motion_module": False,
                                                                                   "unet_use_temporal_attention": False,},).to(accelerator.device)
    # 图像编码器
    image_enc= CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path,).to(device=accelerator.device,dtype=weight_dtype)

    # 用realisedance的poseguider
    pose_guider=get_poseguider(image_finetune=True,num_conds=4).to(device=accelerator.device)  # 第一阶段image_finetune为true




    # 下面规定哪些参数是要训练的，哪些参数是要冻结的
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)

    denoising_unet.requires_grad_(True) # 去噪的参数是要训练的

    for name,param in reference_unet.named_parameters():
        if "up_block.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    pose_guider.requires_grad_(True)

    do_classifier_free_guidance=False

    reference_control_writer= ReferenceAttentionControl(reference_unet,do_classifier_free_guidance=do_classifier_free_guidance,mode="write",fusion_blocks="full",)
    reference_control_reader= ReferenceAttentionControl(denoising_unet,do_classifier_free_guidance=do_classifier_free_guidance,mode="read",fusion_blocks="full",)

    net=Net(reference_unet,denoising_unet,pose_guider,reference_control_writer,reference_control_reader)


    # 看是不是要用内存高效注意力机制
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
    # 判断是否启用梯度检查点
    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    # 将学习率设置成这些因素的累积可以从一定限度上保证在分布式的设置下学习率是合理的
    if cfg.solver.scale_lr:
        print("here")
        learning_rate=(cfg.solver.learning_rate* cfg.solver.gradient_accumulation_steps* cfg.train_bs* accelerator.num_processes)
    else:
        learning_rate=cfg.solver.learning_rate



    # 初始化优化器
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls=bnb.optim.AdamW8bit
    else:
        optimizer_cls=torch.optim.AdamW

    trainable_parms=list(filter(lambda p:p.requires_grad,net.parameters()))
    optimizer=optimizer_cls(trainable_parms,lr=learning_rate,
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
        num_training_steps=cfg.solver.max_train_steps*cfg.solver.gradient_accumulation_steps,
    )

    print("lr_scheduler",lr_scheduler.get_last_lr()[0])
    # 创建tensorboard的writer
    # writer=SummaryWriter(log_dir=os.path.join(save_dir,"logs")) if accelerator.is_local_main_process else None

    # 这里是需要修改的，我们只是暂时这样写罢了
    train_dataset=MyDataset(**cfg.data,is_image=True) # 也就是这里是按照图片来进行训练的

    train_dataloader=torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=cfg.train_bs,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 drop_last=True,
                                                 collate_fn=collate_fn,
                                                 )




    # 下面把东西用accelerator包起来
    (net,optimizer,train_dataloader,lr_scheduler)=accelerator.prepare(net,optimizer,train_dataloader,lr_scheduler)

    # 下面来计算每一轮需要迭代多少步，一轮就是走过一遍dataloader，用dataloader的长度/梯度累积的步数即可
    num_update_steps_per_epoch=math.ceil(
        len(train_dataloader)/ cfg.solver.gradient_accumulation_steps
    )

    # 下面来计算一共要多少轮，也就是总的更新步数/每一轮要更新的步数
    num_train_epochs=math.ceil(
        cfg.solver.max_train_steps /num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(cfg.exp_name,)

    # 下面开始训练
    total_batch_size=(cfg.train_bs*accelerator.num_processes*cfg.solver.gradient_accumulation_steps)

    # 向日志中计入一些信息
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_bs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step=0
    first_epoch=0

    # 如果要从检查点还原的话
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir=save_dir

        dirs=os.listdir(resume_dir)
        dirs=[d for d in dirs if d.startswith("checkpoint")]
        # 这里要关注一下checkpoint是怎么命名的
        dirs=sorted(dirs,key=lambda x:int(x.split("-")[1]))
        path=dirs[-1]
        accelerator.load_state(os.path.join(resume_dir,path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch=global_step//num_update_steps_per_epoch # 这里其实是为了计算当前是第几轮
        resume_step=global_step % num_update_steps_per_epoch

    progress_bar=tqdm(range(global_step,cfg.solver.max_train_steps),disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps") # 设置一下描述文本

    for epoch in range (first_epoch,num_train_epochs):
        train_loss=0.0
        for step,batch in enumerate(train_dataloader):

            with accelerator.accumulate(net):
                pixel_values=batch["pixel_values"].to(device=accelerator.device,dtype=weight_dtype) # 这是原始视频帧

                # 把视频帧放到vae编码到潜在空间中
                with torch.no_grad():
                    latents=vae.encode(pixel_values).latent_dist.sample() # 先映射到潜在表示
                    latents=latents.unsqueeze(2) # [bs,c,h,w]->[bs,c,1,h,w]
                    latents=latents*0.18215

                noise=torch.randn_like(latents) # 生成原始的noise
                if cfg.noise_offset>0.0:
                    noise+=cfg.noise_offset*torch.randn((noise.shape[0],noise.shape[1],1,1,1),device=noise.device)

                bsz=latents.shape[0]

                #timesteps
                timesteps=torch.randint(0,train_noise_scheduler.num_train_timesteps,(bsz,),device=latents.device)

                timesteps=timesteps.long()

                # ref_image_mesh,
                #                 ref_image_keypoint,
                #                 pose_image_mesh,
                #                 pose_image_keypoint,
                pose_img_mesh=batch["pixel_values_mesh"]  # [bs,c,h,w]
                pose_img_mesh=pose_img_mesh.unsqueeze(2)   # [bs,c,1,h,w]

                pose_img_keypoint=batch["pixel_values_keypoint"]
                pose_img_keypoint=pose_img_keypoint.unsqueeze(2) #中间升上去的维度就是frames

                pose_image_hands=batch["pixel_values_hands"]
                pose_image_hands=pose_image_hands.unsqueeze(2)  # 新增的手部mesh

                pose_image_hdkp=batch["pixel_values_hdkp"]
                pose_image_hdkp=pose_image_hdkp.unsqueeze(2)

                # 这是不需要升维的
                ref_img_mesh=batch["pixel_values_ref_mesh"]
                ref_img_keypoint=batch["pixel_values_ref_keypoint"]
                ref_image_hands=batch["pixel_values_ref_hands"]  # ref_hand

                uncond_fwd=random.random()<cfg.uncond_ratio

                clip_image_list=[]
                ref_img_list=[]


                for batch_idx,(ref_img,clip_img) in enumerate(
                        zip(batch["pixel_values_ref_img"],batch["clip_ref_image"])):
                    if uncond_fwd:
                        clip_image_list.append(torch.zeros_like(clip_img))
                    else:
                        clip_image_list.append(clip_img)
                    ref_img_list.append(ref_img)

                # 这里的处理是ref_img过reference_unet的前期处理
                with torch.no_grad():
                    ref_img=torch.stack(ref_img_list,dim=0).to(dtype=vae.dtype,device=vae.device)
                    ref_img_latents=vae.encode(ref_img).latent_dist.sample()
                    ref_img_latents=ref_img_latents*0.18215

                    clip_img=torch.stack(clip_image_list,dim=0).to(dtype=image_enc.dtype,device=image_enc.device)
                    clip_img_embeds=image_enc(clip_img.to("cuda",dtype=weight_dtype)).image_embeds
                    image_prompt_embeds=clip_img_embeds.unsqueeze(1) # (bs,1,d)


                # 在视频帧上添加噪声，形成加到denoising_unet中的噪声输入
                noisy_latents=train_noise_scheduler.add_noise(latents,noise,timesteps)

                # 这里是选择预测的对象
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {train_noise_scheduler.prediction_type}")


                model_pred=net(noisy_latents,timesteps,ref_img_latents,image_prompt_embeds,pose_img_mesh,pose_img_keypoint,pose_image_hands,pose_image_hdkp)# 这里先欠着，后来检查的时候这里应该是少了一个uncond的参数

                # 单步去噪部分，我们这里暂时不要
                # print(model_pred.device)
                # print(timesteps.device)
                # print(noisy_latents.device)
                #
                # latents_pred=train_noise_scheduler.step(model_pred,train_noise_scheduler.timesteps,noisy_latents,return_dict=False)[0]
                # print(latents_pred.shape)
                # image_test=decode_latents(latents_pred,vae=vae)
                # image_test=torch.from_numpy(image_test)
                # for image in image_test:
                #     image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()
                #     res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
                #     res_image_pil.save(f"/home/shipu/mycode/latents_see/image_{i}.jpg")


                if cfg.snr_gamma==0:
                    loss=F.mse_loss(model_pred.float(),target.float(),reduction="mean")
                else:
                    # 这里的信噪比就是公式中 alpha和beta的比值
                    snr=compute_snr(train_noise_scheduler,timesteps)
                    if train_noise_scheduler.config.prediction_type=="v_prediction":
                        snr=snr+1
                        # 这里一开始缩进写错了，但巧合的是预测类型就是v_prediction，现在把缩进修改正确了
                    mse_loss_weights=(torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)],dim=1).min(dim=1)[0]/snr)
                    loss=F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss=(loss.mean(dim=list(range(1,len(loss.shape))))*mse_loss_weights)
                    loss=loss.mean()

                # 收集所有的损失
                avg_loss=accelerator.gather(loss.repeat(cfg.train_bs)).mean()
                train_loss+=avg_loss.item()/cfg.solver.gradient_accumulation_steps

                # 反向传播
                accelerator.backward(loss)
                # 在梯度同步时，对梯度进行裁剪以防梯度爆炸
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parms,cfg.solver.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients: # 这里的if是一种流程上的判断，只有我完成进程同步之后才进行下面的动作
                reference_control_reader.clear()
                reference_control_writer.clear()

                progress_bar.update(1)
                global_step+=1
                accelerator.log({"train_loss":train_loss},step=global_step)
                # writer.add_scalar("train_loss",train_loss,global_step)
                # 在这里我们补了一下，损失是nan的话就直接退出
                if (math.isnan(train_loss)):
                    print("nan is detected, exiting...")
                    # 在这里把model_pred给打出来
                    print(model_pred)
                    print("____________________________________________________")
                    #noisy_latents, timesteps, ref_img_latents, image_prompt_embeds, ref_img_mesh, ref_img_keypoint, pose_img_mesh, pose_img_keypoint
                    print("noisy_latents",noisy_latents)
                    print("timesteps",timesteps)
                    print("ref_img_latents",ref_img_latents)
                    print("image_prompt_embeds",image_prompt_embeds)
                    print("ref_img_mesh",ref_img_mesh)
                    print("ref_img_keypoint",ref_img_keypoint)
                    print("pose_img_mesh",pose_img_mesh)
                    print("pose_img_keypoint",pose_img_keypoint)
                    exit()
                train_loss=0.0

                # 在完成一个batch的训练之后，global_step就向前更新一下
                # 当global_step是检查点步数的整数倍时，我们就保存一下检查点
                if global_step % cfg.checkpointing_steps==0:
                    if accelerator.is_main_process:
                        save_path=os.path.join(save_dir,f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir,1) # 也就是说文件夹里只保留一个检查点
                        accelerator.save_state(save_path)

                    # 这里是有关验证的部分，我们先不写
                    # if (global_step % cfg.val.validation_steps==0) or (global_step in cfg.val.validation_steps_tuple):
                    #     pass

            # 在进度条后面补一些动态变化的信息
            logs={
                    "step_loss":loss.detach().item(),
                    "lr":lr_scheduler.get_last_lr()[0],}
            progress_bar.set_postfix(**logs)

            if global_step>=cfg.solver.max_train_steps:
                break

        # 在每一轮训练完成后，保存一下模型
        if (epoch+1)% cfg.save_model_epoch_interval==0 and accelerator.is_main_process:
            unwrap_net=accelerator.unwrap_model(net)
            save_checkpoint(model=unwrap_net.reference_unet, save_dir=save_dir, prefix="reference_unet",ckpt_num=global_step, total_limit=3)
            save_checkpoint(model=unwrap_net.denoising_unet, save_dir=save_dir, prefix="denoising_unet",ckpt_num=global_step, total_limit=3)
            save_checkpoint(model=unwrap_net.pose_guider, save_dir=save_dir, prefix="pose_guider", ckpt_num=global_step,total_limit=3)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    # writer.close()



def save_checkpoint(model,save_dir,prefix,ckpt_num,total_limit=None):
    save_path=os.path.join(save_dir,f"{prefix}-{ckpt_num}.pth")

    # 这个pre_fix可以理解为固定抬头，以下一大段代码表明最多保存同一个网络的total_limit个权重

    if total_limit is not None:
        checkpoints=os.listdir(save_dir)
        checkpoints=[d for d in checkpoints if d.startswith(prefix)]
        checkpoints=sorted(checkpoints,key=lambda x:int(x.split("-")[1].split(".")[0]))

        if len(checkpoints)>=total_limit:
            num_to_move=len(checkpoints)-total_limit+1
            removing_checkpoints=checkpoints[0:num_to_move]
            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint=os.path.join(save_dir,removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict=model.state_dict()
    torch.save(state_dict,save_path)

def decode_latents(latents,vae):
    video_length=latents.shape[2]
    latents=1/0.18215*latents
    latents=rearrange(latents,"b c f h w->(b f) c h w")
    video=[]

    for frame_idx in tqdm(range(latents.shape[0])):
        video.append(vae.decode(latents[frame_idx:frame_idx+1]).sample)
    video=torch.cat(video)
    video=rearrange(video,"(b f) c h w->b c f h w",f=video_length)
    video=(video/2+0.5).clamp(0,1)
    video=video.cpu().float().numpy()
    return video

if __name__ =="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="/home/shipu/mycode/train_config/train_1.yaml")
    args=parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    # print(type(config))
    # print(config)
    main(config)
    # accelerator=Accelerator()
    # devices = accelerator.state.num_processes
    # print(devices)


# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file /home/shipu/mycode/default_config.yaml train_1_17.py
# CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file  /home/shipu/mycode/one_gpu_default_config.yaml train_1_12.py

# 这是我的改动！！！！！！！！
