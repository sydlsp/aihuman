import torch
from transformers import CLIPVisionModelWithProjection,CLIPProcessor,CLIPTextModelWithProjection,CLIPTokenizer,CLIPTextModel


# (cfg.image_encoder_path,).to(dtype=weight_dtype, device="cuda")
# image_encoder_1=CLIPVisionModelWithProjection.from_pretrained("/remote-home/share/yfsong/shipu/pretrained_model/image_encoder").to(dtype=torch.float16,device="cuda")
#
# image_encoder_2=CLIPVisionModelWithProjection.from_pretrained("/remote-home/share/yfsong/shipu/pretrained_model/text_encoder").to(dtype=torch.float16,device="cuda")
#
# dict_1=image_encoder_1.state_dict()
# dict_2=image_encoder_2.state_dict()
#
# for key in dict_1.keys():
#     print(torch.allclose(dict_1[key],dict_2[key],atol=1e-5))


# print("ok")

text_encoder=CLIPTextModel.from_pretrained("/remote-home/share/yfsong/shipu/pretrained_model/text_encoder").to(dtype=torch.float16,device="cuda")

tokenizer=CLIPTokenizer.from_pretrained("/remote-home/share/yfsong/shipu/pretrained_model/text_encoder")

text_1="husband"

text_2="married"
text_3="man"

text_input_1 = tokenizer(text_1, return_tensors="pt").to("cuda")
text_input_2=tokenizer(text_2,return_tensors="pt").to("cuda")
text_input_3=tokenizer(text_3,return_tensors="pt").to("cuda")

embedding_1=text_encoder(**text_input_1)
embedding_2=text_encoder(**text_input_2)
embedding_3=text_encoder(**text_input_3)

#print(embedding_1.pooler_output.shape)  # 这是句子的全局表示

embedding_1=embedding_1.pooler_output

embedding_4=0.9*embedding_2.pooler_output+0.1*embedding_3.pooler_output

print(torch.nn.functional.cosine_similarity(embedding_1,embedding_4))

print(torch.nn.functional.cosine_similarity(embedding_2.pooler_output,embedding_4))
print(torch.nn.functional.cosine_similarity(embedding_3.pooler_output,embedding_4))

print(torch.nn.functional.cosine_similarity(embedding_1,embedding_2.pooler_output))
print(torch.nn.functional.cosine_similarity(embedding_1,embedding_3.pooler_output))

# 0.7393