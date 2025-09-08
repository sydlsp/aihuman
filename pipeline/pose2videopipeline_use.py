import os
import random
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))

import cv2
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

    def __init__(self,vae,image_encoder,reference_unet,denoising_unet,pose_guider,
                 scheduler: Union[DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler],
                 image_proj_model=None, tokenizer=None, text_encoder=None,):
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

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True,)
        self.cond_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True)

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

    def prepare_latents(self,batch_size,num_channels_latents,
                        width,height,video_length,dtype,device,generator,latents=None,):
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
            video.append(self.vae.decode(latents[frame_idx: frame_idx + 1]).sample)
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
    def __call__(self,ref_image,pose_mesh,pose_keypoint,pose_hands,pose_hdkp,
                 width,height,video_length,num_inference_steps, guidance_scale,
                 num_images_per_prompt=1,eta:float=0.0,
                 generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
                 output_type: Optional[str] = "tensor",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 **kwargs):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 指定设备，暂时没写
        device=torch.device("cuda")
        do_classifier_free_guidance = guidance_scale > 1.0

        # 准备时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size=1

        # 准备clip_image_embedding
        clip_image = self.clip_image_processor.preprocess(ref_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(device, dtype=self.image_encoder.dtype)).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)

        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states=torch.cat([uncond_encoder_hidden_states,encoder_hidden_states],dim=0)

        reference_control_writer = ReferenceAttentionControl(self.reference_unet,
                                                             do_classifier_free_guidance=do_classifier_free_guidance,
                                                             mode="write",
                                                             batch_size=batch_size,
                                                             fusion_blocks="full")

        reference_control_reader = ReferenceAttentionControl(self.denoising_unet,
                                                             do_classifier_free_guidance=do_classifier_free_guidance,
                                                             mode="read",
                                                             batch_size=batch_size,
                                                             fusion_blocks="full")

        num_channels_latents = self.denoising_unet.in_channels

        latents=self.prepare_latents(batch_size*num_images_per_prompt,num_channels_latents,width,height,video_length,clip_image_embeds.dtype,device,generator)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 准备ref_image_latents
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image, height=height, width=width)  # [bs,c,h,w]
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # [bs,4,h,w]

        # 准备mesh条件
        mesh_cond_tensor_list = []
        for mesh_img in pose_mesh:
            mesh_cond_tensor = self.cond_image_processor.preprocess(mesh_img, height=height, width=width)
            mesh_cond_tensor_list.append(mesh_cond_tensor.unsqueeze(2))

        mesh_cond_tensor = torch.cat(mesh_cond_tensor_list, dim=2).to(device)  # [bs,c,h,w]

        kp_cond_tensor_list = []
        for kp_img in pose_keypoint:
            kp_cond_tensor = self.cond_image_processor.preprocess(kp_img, height=height, width=width)
            kp_cond_tensor_list.append(kp_cond_tensor.unsqueeze(2))
        kp_cond_tensor = torch.cat(kp_cond_tensor_list, dim=2).to(device)

        hands_cond_tensor_list = []
        for hands_img in pose_hands:
            hands_cond_tensor = self.cond_image_processor.preprocess(hands_img, height=height, width=width)
            hands_cond_tensor_list.append(hands_cond_tensor.unsqueeze(2))
        hands_cond_tensor = torch.cat(hands_cond_tensor_list, dim=2).to(device)

        hdkp_cond_tensor_list = []
        for hdkp_img in pose_hdkp:
            hdkp_cond_tensor = self.cond_image_processor.preprocess(hdkp_img, height=height, width=width)
            hdkp_cond_tensor_list.append(hdkp_cond_tensor.unsqueeze(2))
        hdkp_cond_tensor = torch.cat(hdkp_cond_tensor_list, dim=2).to(device)
        # print("_________________________________________________________________")
        # print(len(hdkp_cond_tensor_list))
        # print("mesh_cond_tensor.shape", hdkp_cond_tensor_list[0].shape)
        # hdkp_cond_tensor=torch.cat(hdkp_cond_tensor_list,dim=2)
        # print(hdkp_cond_tensor.shape)
        # exit()

        frame = hands_cond_tensor.shape[2]
        mesh_cond_tensor = rearrange(mesh_cond_tensor, "b c f h w -> (b f) c h w")
        kp_cond_tensor = rearrange(kp_cond_tensor, "b c f h w -> (b f) c h w")
        hands_cond_tensor = rearrange(hands_cond_tensor, "b c f h w -> (b f) c h w")
        hdkp_cond_tensor = rearrange(hdkp_cond_tensor, "b c f h w -> (b f) c h w")
        # 开始往pose_guider里面放
        pose_fea=self.pose_guider(mesh_cond_tensor,kp_cond_tensor,hands_cond_tensor,hdkp_cond_tensor,encoder_hidden_states=encoder_hidden_states,temb=timesteps,video_length=len(pose_mesh))
        pose_fea = [rearrange(pe, "(b f) c h w -> b c f h w", f=frame) for pe in pose_fea]


        if do_classifier_free_guidance:
            for idxx in range(len(pose_fea)):
                pose_fea[idxx]=torch.cat([pose_fea[idxx]]*2)

        # 去噪循环
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.denoising_unet(
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

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        images=self.decode_latents(latents)

        if output_type=="tensor":
            images=torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)

from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from diffusers import AutoencoderKL,DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from AniPortrait.src.models.unet_2d_condition import UNet2DConditionModel
from AniPortrait.src.models.unet_3d import UNet3DConditionModel
from redance.poseguider_4cond import get_poseguider
if __name__=="__main__":

    device="cuda"

    infer_config_path = "/home/shipu/mycode/AniPortrait/configs/inference/inference_v2.yaml"  # 这里是为带motion module的Uent需要的一些配置文件\

    infer_config=OmegaConf.load(infer_config_path)

    cfg=OmegaConf.load("/home/shipu/mycode/train_config/train_2.yaml")

    vae=AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device)

    image_enc=CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(device)

    reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder="unet").to(device=device)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path, cfg.mm_path, subfolder="unet",unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs)).to(device=device)
    pose_guider = get_poseguider(image_finetune=False, num_conds=4).to(device=device)

    # 加载训练好的权重
    reference_unet.load_state_dict(torch.load("/data/shipu/ani_train_ckpt/stage2_80000/reference_unet-80000.pth",map_location="cpu"), strict=False,)

    pose_guider.load_state_dict(
        torch.load("/data/shipu/ani_train_ckpt/stage2_80000/pose_guider-80000.pth",map_location="cpu",), strict=False)

    # 加载denoising_unet的时候要分两步加

    denoising_unet.load_state_dict(torch.load("/data/shipu/ani_train_ckpt/stage2_80000/denoising_unet-80000.pth", map_location="cpu"),strict=False)  # 加基础模块

    # 加载scheduler
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)

    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            # timestep_spacing="trailing",
            timestep_spacing="leading",
            prediction_type="v_prediction",
        )

    scheduler = DDIMScheduler(**sched_kwargs)

    pipeline=Pose2VideoPipeline(vae=vae,image_encoder=image_enc,reference_unet=reference_unet,denoising_unet=denoising_unet,pose_guider=pose_guider,scheduler=scheduler)

    for t in range(30,40):
        clip_length=32

        ref_image_path="/data/shipu/train_data_full/train_data_hdkp/videos/video_116_high_quanlity_clip_15/video_116_high_quanlity_clip_15_frame_0000.jpg" # 参考图像的路径

        # /data/shipu/train_data_full/train_data_hdkp/videos/video_194_high_quanlity_clip_9/video_194_high_quanlity_clip_9_frame_0000.jpg
        # /data/shipu/train_data_full/train_data_hdkp/videos/video_116_high_quanlity_clip_15/video_116_high_quanlity_clip_15_frame_0000.jpg
        # /data/shipu/train_data_full/train_data_hdkp/videos/video_215_high_quanlity_clip_68/video_215_high_quanlity_clip_68_frame_0000.jpg
        # in/data/shipu/tra_data_full/train_data_hdkp/videos/video_228_high_quanlity_clip_39/video_228_high_quanlity_clip_39_frame_0000.jpg
        # /data/shipu/train_data_full/train_data_hdkp/videos/video_231_high_quanlity_clip_62/video_231_high_quanlity_clip_62_frame_0000.jpg
        #/data/shipu/train_data_full/train_data_hdkp/videos/video_3_high_quanlity_clip_84/video_3_high_quanlity_clip_84_frame_0000.jpg

        ref_image=cv2.cvtColor(cv2.imread(ref_image_path),cv2.COLOR_BGR2RGB)

        motion_lib_path="/data/shipu/motion_lib_high" # 动作库的路径

        motion_lib=os.listdir(motion_lib_path)

        motion_num=len(motion_lib)

        choose_motion=random.randint(0,motion_num-1)
        # pose_folder_path="/data/shipu/train_data_full/train_data_hdkp/videos/video_215_high_quanlity_clip_68" # 动作文件夹的路径

        pose_folder_path=os.path.join(motion_lib_path,motion_lib[choose_motion])

        # pose_mesh_list=[file for file in os.listdir(pose_folder_path) if "mesh" in file]  // 在这里motion_lib_path里面的文件夹是没有mesh的

        pose_keypoint_list=[file for file in os.listdir(pose_folder_path) if "keypoint" in file]

        pose_hands_list=[file for file in os.listdir(pose_folder_path) if "_full_" in file]

        pose_hdkp_list=[file for file in os.listdir(pose_folder_path) if "hdkp" in file]

        # 在这里我们读取面部mesh视频
        mesh_video_path="/home/shipu/DiffSpeaker/demo_output/diffusion_bias/diffspeaker_wav2vec2_biwi/samples_2025-04-10-19-44-04/speech_long_F3.mp4"

        cap=cv2.VideoCapture(mesh_video_path)
        sample_mesh_img = []
        video_count=0
        while True:
            ret,frame=cap.read()
            video_count+=1

            if video_count==64:  # 也就是读取视频的最大帧为64
                break
            if not ret:
                break

            frame_grb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            sample_mesh_img.append(frame_grb)

        cap.release()  # 释放掉资源



        # 对读取的列表排个序
        def extract_number(filename):
            return int(filename.split(".")[0].split("_")[-1])

        # pose_mesh_list=sorted(pose_mesh_list,key=extract_number)
        pose_keypoint_list=sorted(pose_keypoint_list,key=extract_number)
        pose_hands_list=sorted(pose_hands_list,key=extract_number)
        pose_hdkp_list=sorted(pose_hdkp_list,key=extract_number)


        sample_keypoint_img=[cv2.cvtColor(cv2.imread(os.path.join(pose_folder_path,keypoint)),cv2.COLOR_BGR2RGB) for keypoint in pose_keypoint_list]
        sample_hands_img=[cv2.cvtColor(cv2.imread(os.path.join(pose_folder_path,hands)),cv2.COLOR_BGR2RGB) for hands in pose_hands_list]
        sample_hdkp_img=[cv2.cvtColor(cv2.imread(os.path.join(pose_folder_path,hdkp)),cv2.COLOR_BGR2RGB) for hdkp in pose_hdkp_list]


        ref_image_pil=Image.fromarray(ref_image).convert("RGB")

        mesh_images=[Image.fromarray(mesh).convert("RGB") for mesh in sample_mesh_img]
        keypoint_images=[Image.fromarray(keypoint).convert("RGB") for keypoint in sample_keypoint_img]
        hands_images=[Image.fromarray(hands).convert("RGB") for hands in sample_hands_img]
        hdkp_images=[Image.fromarray(hdkp).convert("RGB") for hdkp in sample_hdkp_img]

        height,width=512,512

        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )

        mesh_tensor_list=[]
        for mesh_image_pil in mesh_images:
            mesh_tensor_list.append(pose_transform(mesh_image_pil))

        keypoint_tensor_list=[]
        for keypoint_image_pil in keypoint_images:
            keypoint_tensor_list.append(pose_transform(keypoint_image_pil))

        hands_tensor_list=[]
        for hands_image_pil in hands_images:
            hands_tensor_list.append(pose_transform(hands_image_pil))

        hdkp_tensor_list=[]
        for hdkp_image_pil in hdkp_images:
            hdkp_tensor_list.append(pose_transform(hdkp_image_pil))

        mesh_tensor=torch.stack(mesh_tensor_list,dim=0)
        mesh_tensor=mesh_tensor.transpose(0,1)

        keypoint_tensor=torch.stack(keypoint_tensor_list,dim=0)
        keypoint_tensor=keypoint_tensor.transpose(0,1)

        hands_tensor=torch.stack(hands_tensor_list,dim=0)
        hands_tensor=hands_tensor.transpose(0,1)

        hdkp_tensor=torch.stack(hdkp_tensor_list,dim=0)
        hdkp_tensor=hdkp_tensor.transpose(0,1)

        # 这才是要放到pipeline中的
        video_length = min(32, len(sample_mesh_img))
        mesh_list=mesh_images[:video_length]
        keypoint_list=keypoint_images[:video_length]
        hands_list=hands_images[:video_length]
        hdkp_list=hdkp_images[:video_length]

        generator=torch.manual_seed(66)

        """
         def __call__(self,ref_image,pose_mesh,pose_keypoint,pose_hands,pose_hdkp,
                     width,height,video_length,num_inference_steps, guidance_scale,
                     num_images_per_prompt=1,eta:float=0.0,
                     generator:Optional[Union[torch.Generator,List[torch.Generator]]]=None,
                     output_type: Optional[str] = "tensor",
                     return_dict: bool = True,
                     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                     callback_steps: Optional[int] = 1,
                     **kwargs):
        """
        pipeline_output=pipeline(ref_image_pil,mesh_list,keypoint_list,hands_list,hdkp_list,width,height,video_length,50,1.5,generator=generator)

        video=pipeline_output.videos

        output_path=f"/home/shipu/mycode/video_output/video_202504010_{t}.mp4"
        save_videos_grid(video,output_path,n_rows=1,fps=8)


# CUDA_VISIBLE_DEVICES=5 python pose2videopipeline_use.py














