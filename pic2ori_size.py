from PIL import Image
import os

# 假设生成的图像是 img，目标大小是 original_width 和 original_height

img_dir="/remote-home/share/yfsong/shipu/test_output/save_5"
img_name_list=os.listdir(img_dir)

save_dir="/remote-home/share/yfsong/shipu/test_output/save_5_ori_size"
os.makedirs(save_dir,exist_ok=True)

for img_name in img_name_list:
    original_image_path=os.path.join(img_dir,img_name)
    img=Image.open(original_image_path)

    print(f"Original Image Size: {img.size}") #输出原始图像的尺寸


    original_width, original_height = 1024,1024
    img_resized = img.resize((original_width, original_height), Image.BILINEAR) # 将其恢复到我们想要的大小
    img_resized.save(os.path.join(save_dir,img_name))
