import copy
from typing import Union,Optional,List,Callable
import os
import sys

"""
在这里我们重写一下怎么用推理管道的过程，之前写的感觉有点问题，在这个版本中，我们固定ref_img,
pose_mesh以及pose_keypoint改变，逐帧生成图片

这个版本是mesh_fea+kp_fea+hand_fea=pose_fea的版本，这是一个试验的推理版本，看看效果怎么样

"""
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__,"..")))

from dataclasses import dataclass
import inspect
from einops import rearrange
from tqdm import tqdm
import numpy as np

import torch
from diffusers import DiffusionPipeline


from diffusers import AutoencoderKL,DDIMScheduler
from AniPortrait.src.models.unet_2d_condition import UNet2DConditionModel
from AniPortrait.src.models.unet_3d import UNet3DConditionModel
# from my_module.new_poseguider import new_poseguider_3 as new_poseguider
from AniPortrait.src.models.pose_guider import PoseGuider as new_poseguider
from train_1_4 import Net
from transformers import CLIPVisionModelWithProjection
from omegaconf import OmegaConf



from diffusers import (DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput,is_accelerate_available
from transformers import CLIPImageProcessor

from AniPortrait.src.models.mutual_self_attention import ReferenceAttentionControl

import cv2
from PIL import Image


@dataclass
class Pose2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class Pose2ImagePipeline(DiffusionPipeline):
    """"""
    _optional_components = []

    def __init__(self,vae,image_encoder,reference_unet,denoising_unet,pose_guider,
                 scheduler:Union[
                        DDIMScheduler,
                        DPMSolverMultistepScheduler,
                        EulerAncestralDiscreteScheduler,
                        EulerDiscreteScheduler,
                        LMSDiscreteScheduler,
                        PNDMScheduler]):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )  # 在这里统一注册一下模块

        self.vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1)  # vae缩放因子
        self.clip_image_processor=CLIPImageProcessor()
        # VaeImageProcessor我们用它的preprocess方法是为了将图像归一化到[-1,1]之间并且大小也是我们想要的
        self.ref_image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,do_convert_rgb=True)
        self.cond_image_processor=VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,do_convert_rgb=True,do_normalize=True)  # 这里为什么要用processor不和之前一样用自己的呢


    # 下面两个决定了是否使用vae的切片功能，启用的话可以在内存有限的情况下处理更大的图像
    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self,gpu_id=0):
        """"""

    def prepare_latents(self,batch_size,num_channels_latents,width,height,dtype,device,generator,latents=None):
        shape=(batch_size,num_channels_latents,height//self.vae_scale_factor,width//self.vae_scale_factor)

        # 这里为什么要那么多生成器，这里的generator应该是随机数的种子
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents=randn_tensor(shape,generator=generator,device=device,dtype=dtype)
        else:
            latents=latents.to(device)

        latents=latents*self.scheduler.init_noise_sigma
        return latents


    def prepare_extra_step_kwargs(self,generator,eta):
        # 准备额外的步骤参数，将这些参数传递给scheduler.step

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

    @property  # 将方法像调用属性一样调用
    def _execution_device(self):
        if self.device !=torch.device("meta") or not hasattr(self.unet,"_hf_hook"):
            return self.device

        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self,latents):
        video_length=latents.shape[2]
        latents=1/0.18215*latents
        latents=rearrange(latents,"b c f h w->(b f) c h w")
        video=[]

        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video=torch.cat(video)
        video=rearrange(video,"(b f) c h w->b c f h w",f=video_length)
        video=(video/2+0.5).clamp(0,1)
        video=video.cpu().float().numpy()
        return video

    @torch.no_grad()
    def __call__(self,
                 ref_image,  # 参考图
                 # pose_image,
                 # ref_pose_image,
                 pose_mesh,
                 pose_keypoint,
                 pose_hand, # 手部的关键点
                 ref_mesh,
                 ref_keypoint,
                 ref_hand, # 参考图像的手部关键点
                 width,
                 height,
                 num_inference_steps,
                 guidance_scale,
                 num_images_per_prompt=1,
                 eta:float=0.0,
                 generator:Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 output_type: Optional[str] = "tensor",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: Optional[int] = 1,
                 **kwargs,
                 ):

        """"""
        height=height or self.unet.config.sample_size *self.vae_scale_factor  # 这里的写法是有问题的其实应该是self.denoising_unet ...
        width=width or self.unet.config.sample_size *self.vae_scale_factor

        print("________________________________________________________________________")
        print(self.scheduler.config.prediction_type)

        device=torch.device("cuda:2")
        #device= self._execution_device  # 这里修改了一下记得还原

        do_classifier_free_guidance=guidance_scale>1.0

        # 准备timesteps
        self.scheduler.set_timesteps(num_inference_steps,device=device)
        timesteps=self.scheduler.timesteps

        batch_size=1


        # 准备 clip_image_embeds
        clip_image=self.clip_image_processor.preprocess(ref_image.resize((224,224)),return_tensors="pt").pixel_values
        clip_image_embeds=self.image_encoder(clip_image.to(device,dtype=self.image_encoder.dtype)).image_embeds
        image_prompt_embeds=clip_image_embeds.unsqueeze(1)

        uncond_image_prompt_embeds=torch.zeros_like(image_prompt_embeds)
        # 如果做cfg的话
        if do_classifier_free_guidance:
            image_prompt_embeds=torch.cat([uncond_image_prompt_embeds,image_prompt_embeds],dim=0)

        reference_control_writer=ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        reference_control_reader=ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents=self.denoising_unet.in_channels

        # 准备latents，这里返回的是[batch_size,num_channels_latents,height//vae_scale,width//vae_scale]的高斯噪声
        latents=self.prepare_latents(batch_size*num_images_per_prompt,num_channels_latents,width,height,clip_image_embeds.dtype,device,generator)
        latents=latents.unsqueeze(2) # 把帧维度补充上，这里是单张图像，所以直接补个1就行了
        latents_dtype=latents.dtype

        # 准备额外的参数
        extra_step_kwargs=self.prepare_extra_step_kwargs(generator,eta)

        # 准备好ref_image_latents
        ref_image_tensor=self.ref_image_processor.preprocess(ref_image,height=height,width=width) # (bs,c,h,w) 这里调用的是vae处理过程
        ref_image_tensor=ref_image_tensor.to(dtype=self.vae.dtype,device=self.vae.device)
        ref_image_latents=self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents=ref_image_latents*0.18215  # (bs,4,h,w)

        # 准备好条件图像，这里也就是posh_mesh以及pose_keypoint
        mesh_cond_tensor=self.cond_image_processor.preprocess(pose_mesh,height=height,width=width) # 这里还是调用vae预处理过程
        keypoint_cond_tensor=self.cond_image_processor.preprocess(pose_keypoint,height=height,width=width)
        hand_cond_tensor=self.cond_image_processor.preprocess(pose_hand,height=height,width=width)  # 新加的手的

        mesh_cond_tensor=mesh_cond_tensor.unsqueeze(2)
        mesh_cond_tensor=mesh_cond_tensor.to(device=device,dtype=self.pose_guider.dtype)


        keypoint_cond_tensor=keypoint_cond_tensor.unsqueeze(2)
        keypoint_cond_tensor=keypoint_cond_tensor.to(device=device,dtype=self.pose_guider.dtype)

        hand_cond_tensor=hand_cond_tensor.unsqueeze(2)
        hand_cond_tensor=hand_cond_tensor.to(device=device,dtype=self.pose_guider.dtype)

        # 准备好参考图的mesh以及keypoint
        ref_mesh_tensor=self.cond_image_processor.preprocess(ref_mesh,height=height,width=width)
        ref_keypoint_tensor=self.cond_image_processor.preprocess(ref_keypoint,height=height,width=width)

        ref_mesh_tensor=ref_mesh_tensor.to(device=device,dtype=self.pose_guider.dtype)
        ref_keypoint_tensor=ref_keypoint_tensor.to(device=device,dtype=self.pose_guider.dtype)

        """新加的手部关键点"""
        ref_hand_tensor=self.cond_image_processor.preprocess(ref_hand,height=height,width=width)
        ref_hand_tensor=ref_hand_tensor.to(device=device,dtype=self.pose_guider.dtype)

        mesh_fea=self.pose_guider(mesh_cond_tensor,ref_mesh_tensor)
        keypoint_fea=self.pose_guider(keypoint_cond_tensor,ref_keypoint_tensor)
        hand_fea=self.pose_guider(hand_cond_tensor,ref_hand_tensor)


        pose_fea=mesh_fea+keypoint_fea+hand_fea

        # 写到这里也就是说做cfg的时候，不管是有条件还是无条件都是接受pose_pea的
        if do_classifier_free_guidance:
            for idxx in range(len(pose_fea)):
                pose_fea[idxx]=torch.cat([pose_fea[idxx]]*2)

        """
        去噪循环
        """

        # 用总的时间步数去掉去噪步骤的数量得到预热步骤的数量
        num_warmup_steps=len(timesteps)-num_inference_steps*self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                if i==0:
                    self.reference_unet(ref_image_latents.repeat(
                        (2 if do_classifier_free_guidance else 1),1,1,1),
                        torch.zeros_like(t), # 为什么这里要创建一个和t一样的tensor
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                # 如果做cfg的话需要拓展一下latents
                latent_model_input=torch.cat([latents]*2) if do_classifier_free_guidance else latents
                latent_model_input=self.scheduler.scale_model_input(latent_model_input,t) # 这里根据时间步对输入进行缩放，是为了更好的生成
                noise_pred=self.denoising_unet(latent_model_input,t,encoder_hidden_states=image_prompt_embeds,pose_cond_fea=pose_fea,return_dict=False)[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond,noise_pred_text=noise_pred.chunk(2)
                    noise_pred=noise_pred_uncond+guidance_scale*(noise_pred_text-noise_pred_uncond)
                # 根据预测出的噪声和上一步的噪声来去噪
                latents=self.scheduler.step(noise_pred,t,latents,**extra_step_kwargs,return_dict=False)[0]

                # 每10步保存一下，看看图
                # if (i%10==0):
                #     image_process=torch.from_numpy(self.decode_latents(latents))
                #     image_process = image_process[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, h, w)
                #     image_process_pil = Image.fromarray((image_process * 255).astype(np.uint8))
                #     image_process_pil.save(f"/remote-home/yfsong/shipu/mycode/denoising_see/image_{i}.jpg")


                if i==len(timesteps)-1 or ( (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    # 回调函数相关
                    if callback is not None and i % callback_steps==0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        image=self.decode_latents(latents)

        if output_type=="tensor":
            image=torch.from_numpy(image)

        if not return_dict:
            return image

        return Pose2ImagePipelineOutput(images=image)





device="cuda:2"
# 先把要用的基础模型加进来
cfg=OmegaConf.load("/remote-home/yfsong/shipu/mycode/train_config/train_1.yaml")
# 初始化VAE
vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(device)
# 图像编码器
image_enc= CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path,).to(device=device)
# 以上两个不是放到模型中的

# 初始化reference_unet
reference_unet = UNet2DConditionModel.from_pretrained(cfg.base_model_path,subfolder="unet").to(device=device)
#初始化denoising_unet
denoising_unet=UNet3DConditionModel.from_pretrained_2d(cfg.base_model_path,"",subfolder="unet",
                                                       unet_additional_kwargs={"use_motion_module": False,
                                                                               "unet_use_temporal_attention": False,},).to(device=device)



do_classifier_free_guidance=False

reference_control_writer= ReferenceAttentionControl(reference_unet,do_classifier_free_guidance=do_classifier_free_guidance,mode="write",fusion_blocks="full",)
reference_control_reader= ReferenceAttentionControl(denoising_unet,do_classifier_free_guidance=do_classifier_free_guidance,mode="read",fusion_blocks="full",)
pose_guider=new_poseguider().to(device=device)


model=Net(reference_unet,denoising_unet,pose_guider,reference_control_writer,reference_control_reader)


# 这是从bin文件中直接把所有权重包装好的直接拿
ckpt_path="/remote-home/share/yfsong/shipu/test_output/stage1_4/checkpoint-100000/pytorch_model.bin"
ckpt=torch.load(ckpt_path,map_location=device)
model.load_state_dict(ckpt)


# 现在我们加载完了模型，为了用pipeline要把一个个模型给抽出来
reference_unet = model.reference_unet
denoising_unet = model.denoising_unet
pose_guider = model.pose_guider
#
# reference_unet_ckpt_path="/remote-home/share/yfsong/shipu/test_output/stage1_3/reference_unet-4940.pth"
# denoising_unet_ckpt_path="/remote-home/share/yfsong/shipu/test_output/stage1_3/denoising_unet-4940.pth"
# pose_guider_ckpt_path="/remote-home/share/yfsong/shipu/test_output/stage1_3/pose_guider-4940.pth"

# reference_unet.load_state_dict(torch.load(reference_unet_ckpt_path,map_location=device))
# denoising_unet.load_state_dict(torch.load(denoising_unet_ckpt_path,map_location=device))
# pose_guider.load_state_dict(torch.load(pose_guider_ckpt_path,map_location=device))

# vae,image_encoder,reference_unet,denoising_unet,pose_guider,
sched_kwargs=OmegaConf.to_container(cfg.noise_scheduler_kwargs)

if cfg.enable_zero_snr:
    sched_kwargs.update(
        rescale_betas_zero_snr=True,
        # timestep_spacing="trailing",
        timestep_spacing="leading",
        prediction_type="v_prediction",
    )
scheduler=DDIMScheduler(**sched_kwargs)


pipeline=Pose2ImagePipeline(vae=vae,image_encoder=image_enc,reference_unet=reference_unet,denoising_unet=denoising_unet,pose_guider=pose_guider,scheduler=scheduler)


pose_dir="/remote-home/share/yfsong/shipu/dataset_video_true_2/videos/video_14_high_quanlity_clip_20"


pose_keypoint_list=[i for i in os.listdir(pose_dir) if "keypoint" in i]
pose_mesh_list=[i for i in os.listdir(pose_dir) if "mesh" in i]
pose_hand_list=[i for i in os.listdir(pose_dir) if "_hands_" in i]

def extract_number(filename):
    return int(filename.split(".")[0].split("_")[-1])

pose_keypoint_list=sorted(pose_keypoint_list,key=extract_number)
pose_mesh_list=sorted(pose_mesh_list,key=extract_number)
pose_hand_list=sorted(pose_hand_list,key=extract_number)

print(pose_keypoint_list)
print(pose_mesh_list)

# 准备好ref_img的相关图片
ref_frame_path="/remote-home/share/yfsong/shipu/dataset_video_true_2/videos/video_14_high_quanlity_clip_1/video_14_high_quanlity_clip_1_frame_0002.jpg"
ref_mesh_path="/remote-home/share/yfsong/shipu/dataset_video_true_2/videos/video_14_high_quanlity_clip_1/video_14_high_quanlity_clip_1_mesh_0002.jpg"
ref_keypoint_path="/remote-home/share/yfsong/shipu/dataset_video_true_2/videos/video_14_high_quanlity_clip_1/video_14_high_quanlity_clip_1_keypoint_0002.jpg"
ref_hand_path="/remote-home/share/yfsong/shipu/dataset_video_true_2/videos/video_14_high_quanlity_clip_1/video_14_high_quanlity_clip_1_hands_0002.jpg"  # 这里补上hands的路径



ref_frame=cv2.imread(ref_frame_path)
ref_frame=cv2.cvtColor(ref_frame,cv2.COLOR_BGR2RGB)
ref_frame_pil=Image.fromarray(ref_frame).convert('RGB')

ref_mesh=cv2.imread(ref_mesh_path)
ref_mesh=cv2.cvtColor(ref_mesh,cv2.COLOR_BGR2RGB)
ref_mesh_pil=Image.fromarray(ref_mesh).convert('RGB')

ref_keypoint=cv2.imread(ref_keypoint_path)
ref_keypoint=cv2.cvtColor(ref_keypoint,cv2.COLOR_BGR2RGB)
ref_keypoint_pil=Image.fromarray(ref_keypoint).convert('RGB')

ref_hand=cv2.imread(ref_hand_path)
ref_hand=cv2.cvtColor(ref_hand,cv2.COLOR_BGR2RGB)
ref_hand_pil=Image.fromarray(ref_hand).convert('RGB')

# 在这里我们重写一下，一次性生成处理多张图片的生成




generator = torch.manual_seed(100)


# vision_dir="/remote-home/yfsong/shipu/mycode/pic_see"

# for i in range(len(pose_mesh_list)):
for i in range(0,min(25,len(pose_mesh_list))):

    # pose
    pose_mesh_path=os.path.join(pose_dir,pose_mesh_list[i])
    pose_keypoint_path=os.path.join(pose_dir,pose_keypoint_list[i])
    pose_hand_path=os.path.join(pose_dir,pose_hand_list[i])


    pose_mesh=cv2.imread(pose_mesh_path)
    pose_mesh=cv2.cvtColor(pose_mesh,cv2.COLOR_BGR2RGB)
    pose_mesh_pil=Image.fromarray(pose_mesh).convert('RGB')

    pose_keypoint=cv2.imread(pose_keypoint_path)
    pose_keypoint=cv2.cvtColor(pose_keypoint,cv2.COLOR_BGR2RGB)
    pose_keypoint_pil=Image.fromarray(pose_keypoint).convert('RGB')

    pose_hand=cv2.imread(pose_hand_path)
    pose_hand=cv2.cvtColor(pose_hand,cv2.COLOR_BGR2RGB)
    pose_hand_pil=Image.fromarray(pose_hand).convert('RGB')

    # ref_frame_pil.save(os.path.join(vision_dir,f"ref_frame_{i}.jpg"))
    # pose_mesh_pil.save(os.path.join(vision_dir,f"pose_mesh_{i}.jpg"))
    # pose_keypoint_pil.save(os.path.join(vision_dir,f"pose_keypoint_{i}.jpg"))
    # ref_mesh_pil.save(os.path.join(vision_dir,f"ref_mesh_{i}.jpg"))
    # ref_keypoint_pil.save(os.path.join(vision_dir,f"ref_keypoint_{i}.jpg"))

    # 那也就是说现在保证了读进去的数据都是RGB的
    image=pipeline(ref_image=ref_frame_pil,pose_mesh=pose_mesh_pil,pose_keypoint=pose_keypoint_pil,pose_hand=pose_hand_pil,ref_mesh=ref_mesh_pil,ref_keypoint=ref_keypoint_pil,ref_hand=ref_hand_pil,width=cfg.data.sample_size[0],height=cfg.data.sample_size[1],num_inference_steps=250,guidance_scale=3.5,generator=generator).images
    image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, h, w)
    res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
    res_image_pil.save(f"/remote-home/yfsong/shipu/mycode/output/save_20/res_image_{i}.jpg")
    print(f"save image {i}")







