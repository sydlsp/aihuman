import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))

import torchvision.utils

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__,"..")))

from typing import Callable,List,Optional,Union
from dataclasses import dataclass
import numpy as np
import inspect
import torch
from einops import rearrange
from tqdm import tqdm
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput,is_accelerate_available
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import (DDIMScheduler,DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler,LMSDiscreteScheduler,
                                  PNDMScheduler)
from transformers import CLIPImageProcessor

from AniPortrait.src.models.mutual_self_attention import ReferenceAttentionControl
from AniPortrait.src.utils.util import save_videos_grid

@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos:Union[torch.Tensor,np.ndarray]

class Pose2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(self,
                 vae,
                 image_encoder,
                 reference_unet,
                 denoising_unet,
                 pose_guider,
                 scheduler:Union[DDIMScheduler,
                 PNDMScheduler,LMSDiscreteScheduler,EulerDiscreteScheduler,
                 EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler],
                 image_proj_model=None,
                 tokenizer=None,
                 text_encoder=None,):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        self.vae_scale_factor=2**(len(self.vae.config.block_out_channels)-1)
        self.clip_image_processor=CLIPImageProcessor()
        self.ref_image_processor=VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,do_convert_rgb=True,
        )

        self.cond_image_processor=VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,do_convert_rgb=True,do_normalize=True)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        width,
                        height,
                        video_length,
                        dtype,
                        device,
                        generator,
                        latents=None,):
        shape=(batch_size,num_channels_latents,video_length,height//self.vae_scale_factor,width//self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents=latents.to(device)

        latents=latents*self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs



    @torch.no_grad()
    def __call__(self,
                 ref_image,
                 pose_mesh,
                 pose_keypoint,
                 ref_mesh,
                 ref_keypoint,
                 width,
                 height,
                 video_length,
                 num_inference_steps,
                 guidance_scale,
                 num_images_per_prompt=1,
                 eta:float=0.0,
                 generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
                 output_type: Optional[str] = "tensor",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 **kwargs):
        height=height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # device=self._execution_device
        device=torch.device("cuda:3")

        do_classifier_free_guidance = guidance_scale > 1.0

        # 准备时间步
        self.scheduler.set_timesteps(num_inference_steps,device=device)
        timesteps=self.scheduler.timesteps

        batch_size=1

        # 准备clip image embeds
        clip_image=self.clip_image_processor.preprocess(ref_image,return_tensors="pt").pixel_values
        clip_image_embeds=self.image_encoder(clip_image.to(device,dtype=self.image_encoder.dtype)).image_embeds
        encoder_hidden_states=clip_image_embeds.unsqueeze(1)

        uncond_encoder_hidden_states=torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states=torch.cat([uncond_encoder_hidden_states,encoder_hidden_states],dim=0)

        reference_control_writer=ReferenceAttentionControl(self.reference_unet,
                                                           do_classifier_free_guidance=do_classifier_free_guidance,
                                                           mode="write",
                                                           batch_size=batch_size,
                                                           fusion_blocks="full")

        reference_control_reader=ReferenceAttentionControl(self.denoising_unet,
                                                           do_classifier_free_guidance=do_classifier_free_guidance,
                                                           mode="read",
                                                           batch_size=batch_size,
                                                           fusion_blocks="full")

        num_channels_latents=self.denoising_unet.in_channels

        latents=self.prepare_latents(
            batch_size*num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator
        )

        extra_step_kwargs=self.prepare_extra_step_kwargs(generator,eta)

        # 准备好ref_image_latents
        ref_image_tensor=self.ref_image_processor.preprocess(ref_image,height=height,width=width) # [bs,c,h,w]

        ref_image_tensor=ref_image_tensor.to(dtype=self.vae.dtype,device=self.vae.device)

        ref_image_latents=self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents=ref_image_latents*0.18215 # [bs,4,h,w]

        # 在这个pipeline中，要准备图像条件的列表，和之前的pipeline不一样
        # 推断这里应该是由于我们不是pose2pic，而是pose2vid 这样导致动作是一个列表
        mesh_cond_tensor_list=[]
        for mesh_img in pose_mesh:
            mesh_cond_tensor=self.cond_image_processor.preprocess(mesh_img,height=height,width=width).transpose(0,1) # [c,1,h,w]
            mesh_cond_tensor_list.append(mesh_cond_tensor)

        mesh_cond_tensor=torch.cat(mesh_cond_tensor_list,dim=1)

        kp_cond_tensor_list=[]
        for kp_img in pose_keypoint:
            kp_cond_tensor=self.cond_image_processor.preprocess(kp_img,height=height,width=width).transpose(0,1) # [c,1,h,w]
            kp_cond_tensor_list.append(kp_cond_tensor)

        kp_cond_tensor=torch.cat(kp_cond_tensor_list,dim=1)

        mesh_cond_tensor=mesh_cond_tensor.unsqueeze(0)
        kp_cond_tensor=kp_cond_tensor.unsqueeze(0)

        mesh_cond_tensor=mesh_cond_tensor.to(device=device,dtype=self.pose_guider.dtype)
        kp_cond_tensor=kp_cond_tensor.to(device=device,dtype=self.pose_guider.dtype)

        # 参考图像的mesh和kp
        ref_mesh_tensor=self.cond_image_processor.preprocess(ref_mesh,height=height,width=width)
        ref_mesh_tensor=ref_mesh_tensor.to(device=device,dtype=self.pose_guider.dtype)

        ref_keypoint_tensor=self.cond_image_processor.preprocess(ref_keypoint,height=height,width=width)
        ref_keypoint_tensor=ref_keypoint_tensor.to(device=device,dtype=self.pose_guider.dtype)

        # 下面开始往pose_guider里面放
        mesh_fea=self.pose_guider(mesh_cond_tensor,ref_mesh_tensor)
        kp_fea=self.pose_guider(kp_cond_tensor,ref_keypoint_tensor)

        pose_fea=mesh_fea+kp_fea

        if do_classifier_free_guidance:
            for idxx in range(len(pose_fea)):
                pose_fea[idxx]=torch.cat([pose_fea[idxx]]*2)

        """
        去噪循环
        """
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                if i==0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1),1,1,1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                latent_model_input=(
                    torch.cat([latents]*2) if do_classifier_free_guidance else latents
                )
                latent_model_input=self.scheduler.scale_model_input(latent_model_input,t)

                noise_pred=self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    pose_cond_fea=pose_fea,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                # 这里准备一下，再看看，这儿是完全复制的
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        images=self.decode_latents(latents)  # 这里是视频的解码

        if output_type=="tensor":
            images=torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)

import cv2
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from diffusers import AutoencoderKL,DDIMScheduler
from transformers import CLIPVisionModelWithProjection

from AniPortrait.src.models.unet_2d_condition import UNet2DConditionModel
from AniPortrait.src.models.unet_3d import UNet3DConditionModel
from AniPortrait.src.models.pose_guider import PoseGuider as new_poseguider

if __name__=="__main__":
    """
      这里我们把推理过程写在main函数里
    """
    device="cuda:3"

    infer_config_path="/remote-home/yfsong/shipu/mycode/AniPortrait/configs/inference/inference_v2.yaml" # 这里是为带motion module的Uent需要的一些配置文件

    infer_config=OmegaConf.load(infer_config_path)

    cfg=OmegaConf.load("/remote-home/yfsong/shipu/mycode/train_config/train_2.yaml")

    vae=AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device)

    image_enc=CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(device)
    # 以上两个模型不放到Net中


    # 只有这三个模型是我们要的
    reference_unet=UNet2DConditionModel.from_pretrained(cfg.base_model_path,subfolder="unet").to(device=device)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path,cfg.mm_path,subfolder="unet",unet_additional_kwargs=OmegaConf.to_container(
            infer_config.unet_additional_kwargs)).to(device=device)
    pose_guider=new_poseguider().to(device=device)

    # 注意，在这里pose2imgpipe中我们写了ReferenceAttentionControl，实际上并没有用到，在那里写只是为了加载原始模型的参数用到的
    # 在这里我们不用Net然后再把模型取出来，我们直接就用模型就可以了

    # 加载训练好的权重
    stage1_ckpt_dir=cfg.stage1_ckpt_dir
    stage1_ckpt_step=cfg.stage1_ckpt_step

    reference_unet.load_state_dict(torch.load(os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),strict=False,)

    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
    ),strict=False)

    # 加载denoising_unet的时候要分两步加

    denoising_unet.load_state_dict(
        torch.load(os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"), map_location="cpu"),
        strict=False
    )  # 加基础模块

    denoising_unet.load_state_dict(
        torch.load("/remote-home/share/yfsong/shipu/test_output/stage2/checkpoint-40000/pytorch_model.bin", map_location="cpu"),
        strict=False,
    )  # 加motion module的权重，这里就手动写地址

    # 截止以上，模型权重是加载成功了

    # 加载scheduler
    sched_kwargs=OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            # timestep_spacing="trailing",
            timestep_spacing="leading",
            prediction_type="v_prediction",
        )
    scheduler=DDIMScheduler(**sched_kwargs)

    pipeline=Pose2VideoPipeline(vae=vae,image_encoder=image_enc,reference_unet=reference_unet,denoising_unet=denoising_unet,pose_guider=pose_guider,scheduler=scheduler)

    clip_length=32  # 理想的视频片段的长度


    # 下面重点关注一下怎么读数据，这里看一下Dateset和DatasetValid有什么区别
    # 主要区别就是DatasetValid中不要ref_img的clip图像和pose相关的图像
    # 从理论上来讲，生成video和生成image是一样的，没什么大的区别

    # 我们继续仿照train_stage_2中的代码来写，先cv2读数据，然后cvtColor，再Image.fromarray
    # 我们把这个过程写成一个函数 image2PIL


    # 先别急，一步一步慢慢写

    ref_image_path="/remote-home/share/yfsong/shipu/dataset_video_true_1/videos/video_10_high_quanlity_clip_0/video_10_high_quanlity_clip_0_frame_0000.jpg"
    ref_mesh_path="/remote-home/share/yfsong/shipu/dataset_video_true_1/videos/video_10_high_quanlity_clip_0/video_10_high_quanlity_clip_0_mesh_0000.jpg"
    ref_keypoint_path="/remote-home/share/yfsong/shipu/dataset_video_true_1/videos/video_10_high_quanlity_clip_0/video_10_high_quanlity_clip_0_keypoint_0000.jpg"

    sample_ref_img=cv2.cvtColor(cv2.imread(ref_image_path),cv2.COLOR_BGR2RGB)  #ref_image是图片，这里我们的命名规则和train_stage_2中的sample['ref_img']保持一致
    sample_ref_mesh=cv2.cvtColor(cv2.imread(ref_mesh_path),cv2.COLOR_BGR2RGB)
    sample_ref_keypoint=cv2.cvtColor(cv2.imread(ref_keypoint_path),cv2.COLOR_BGR2RGB)

    pose_folder_path = "/remote-home/share/yfsong/shipu/dataset_video_true_1/videos/video_10_high_quanlity_clip_0"  # 这里是存放posh_mesh和pose_keypoint的文件夹地址

    len_folder = len(os.listdir(pose_folder_path)) // 3  # 防止取超过索引

    pose_mesh_list = [file for file in os.listdir(pose_folder_path) if "mesh" in file][:min(clip_length, len_folder)]
    pose_keypoint_list = [file for file in os.listdir(pose_folder_path) if "keypoint" in file][:min(clip_length, len_folder)]

    # 对读取的列表排个序
    def extract_number(filename):
        return int(filename.split(".")[0].split("_")[-1])

    pose_mesh_list=sorted(pose_mesh_list,key=extract_number)
    pose_keypoint_list=sorted(pose_keypoint_list,key=extract_number)

    # mesh_img和keypoint_img要组合成列表
    sample_mesh_img=[cv2.cvtColor(cv2.imread(os.path.join(pose_folder_path,mesh)),cv2.COLOR_BGR2RGB) for mesh in pose_mesh_list]
    sample_keypoint_img=[cv2.cvtColor(cv2.imread(os.path.join(pose_folder_path,keypoint)),cv2.COLOR_BGR2RGB) for keypoint in pose_keypoint_list]


    # 从此开始写法和train_stage_2中的类似
    ref_image_pil=Image.fromarray(sample_ref_img).convert("RGB")
    mesh_images=[Image.fromarray(mesh).convert("RGB") for mesh in sample_mesh_img]
    keypoint_images=[Image.fromarray(keypoint).convert("RGB") for keypoint in sample_keypoint_img]

    # 到上面数据格式是没什么问题的
    height,width=256,256

    pose_transform=transforms.Compose(
        [transforms.Resize((height,width)),transforms.ToTensor()]
    )

    mesh_tensor_list=[]
    keypoint_tensor_list=[]
    ref_tensor_list=[]

    for mesh_image_pil in mesh_images:
        mesh_tensor_list.append(pose_transform(mesh_image_pil))
        ref_tensor_list.append(pose_transform(ref_image_pil))  # 这里ref_image也是同步在加进去

    for keypoint_image_pil in keypoint_images:
        keypoint_tensor_list.append(pose_transform(keypoint_image_pil))


    mesh_tensor=torch.stack(mesh_tensor_list,dim=0)
    mesh_tensor=mesh_tensor.transpose(0,1)

    keypoint_tensor=torch.stack(keypoint_tensor_list,dim=0)
    keypoint_tensor=keypoint_tensor.transpose(0,1)

    ref_tensor=torch.stack(ref_tensor_list,dim=0)
    ref_tensor=ref_tensor.transpose(0,1)


    # 下面的才是我们要放进到pipeline中的

    mesh_list=sample_mesh_img
    keypoint_list=sample_keypoint_img

    ref_mesh=sample_ref_mesh
    ref_keypoint=sample_ref_keypoint

    """
                 ref_image,
                 pose_mesh,
                 pose_keypoint,
                 ref_mesh,
                 ref_keypoint,
                 width,
                 height,
                 video_length,
                 num_inference_steps,
                 guidance_scale,
                 num_images_per_prompt=1,
                 eta:float=0.0,
                 generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
                 output_type: Optional[str] = "tensor",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 **kwargs):
    """
    generator=torch.manual_seed(42)

    video_length=min(clip_length,len_folder)

    pipeline_output=pipeline(ref_image_pil,mesh_list,keypoint_list,ref_mesh,ref_keypoint,width,height,video_length,250,3.5,generator=generator)

    video=pipeline_output.videos

    print(type(video))

    # 到这里是能跑的，代码上没有太大问题，明天看下生成结果和逻辑有没有问题

    # 下面我们继续写保存成gif的代码
    # 在源代码中 video是和[ref_tensor,pose_tensor,video,gt_tensor] cat 在dim=0 保存的
    # 由于是在dim=0 cat 其实只是改变了batch_size的大小，里面的数据维度是没变的,所以还是可以直接调用
    output_path="/remote-home/share/yfsong/shipu/test_output/stage2/show_video_1.gif"
    save_videos_grid(video,output_path,n_rows=1)  #fps默认是8




























    # def image2PIL(image_path):
    #     image=cv2.imread(image_path)
    #     image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #     image=Image.fromarray(image).convert('RGB')
    #     return image
    #
    # # 上面是下面四行代码的函数表示
    # # ref_image_path="ref_image_path"
    # # ref_image=cv2.imread(ref_image_path)
    # # ref_image=cv2.cvtColor(ref_image,cv2.COLOR_BGR2RGB)
    # # ref_image=Image.fromarray(ref_image).convert('RGB')
    #
    # ref_image_path="ref_image_path"
    # ref_image=image2PIL(ref_image_path)
    #
    # # 同样的，这里我们要读pose_mesh和pose_kp
    # pose_folder_path="pose_folder_path"  # 这里是存放posh_mesh和pose_keypoint的文件夹地址
    #
    # len_folder=len(os.listdir(pose_folder_path))//3  # 防止取超过索引
    #
    # pose_mesh_list=[file for file in os.listdir(pose_folder_path) if "mesh" in file][:min(clip_length,len_folder)]
    # pose_keypoint_list=[file for file in os.listdir(pose_folder_path) if "keypoint" in file][:min(clip_length,len_folder)]
    #
    # # 我们还是和之前一样，对文件名进行排序
    # def extract_number(filename):
    #     return int(filename.split(".")[0].split("_")[-1])
    #
    # pose_mesh_list=sorted(pose_mesh_list,key=extract_number)
    # pose_keypoint_list=sorted(pose_keypoint_list,key=extract_number)
    #
    # mesh_images=[image2PIL(os.path.join(pose_folder_path,mesh)) for mesh in pose_mesh_list]
    # keypoint_images=[image2PIL(os.path.join(pose_folder_path,keypoint)) for keypoint in pose_keypoint_list]
    #
    # # 下面要搞懂一个事情tar_gt到底是什么，感觉是真实值，也就是真实的视频帧是什么样的，这里先不管
    # tar_gt=None  # 到时候补充
    #
    #
    # height,width=256,256
    #
    # pose_transform=transforms.Compose(
    #     [transforms.Resize((height,width)),transforms.ToTensor()]
    # )
    #
    #
    # mesh_tensor_list=[]
    # keypoint_tensor_list=[]
    # ref_tensor_list=[]
    # """
    # """
    # for mesh_pil in mesh_images: # 在前面我们已经规定好数据最长就是clip_legth
    #     mesh_tensor_list.append(pose_transform(mesh_pil))
    #     # 看一下，这里的ref_tensor也要同步走一下
    #     ref_tensor_list.append(pose_transform(ref_image))
    #
    # for keypoint_pil in keypoint_images:
    #     keypoint_tensor_list.append(pose_transform(keypoint_pil))






























