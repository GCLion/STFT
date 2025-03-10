import os
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from ldm.models.diffusion.ddpm import LatentDiffusion
import warnings
warnings.filterwarnings("ignore")

def avg_flatten(z_t):
    z_t_pooled = F.adaptive_avg_pool2d(z_t, (8, 8))
    # 展平为一维张量
    z_t_flattened = z_t_pooled.view(-1)  # 变为 3*16*16 的一维张量
    return z_t_flattened

def processImg(image_path, model):
    batch_size = 20
    # channels = 3
    # image_size = 64
    ddim_steps = 20
    ddim_eta = 1.0

    image = Image.open(image_path).convert("RGB")
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to("cuda")
    samples, intermediates = model.sample_log(cond=image_tensor, batch_size=batch_size, ddim=True, 
                        ddim_steps=ddim_steps, ddim_eta=ddim_eta)
    z_t_list = [avg_flatten(z_t) for _, z_t in enumerate(samples)]
    Z = torch.stack(z_t_list, dim=0)
    print(Z.shape)
    return Z

def get_features(model, sub_dir, classes):
    x = torch.tensor([1, 2]).cuda()
    root_dir = '/data/usr/lhr/GenImage/' + sub_dir + "val"
    tar_dir = os.path.join(root_dir, classes)
    # print(tar_dir) # /data/usr/lhr/GenImage/ADM/imagenet_ai_0508_adm/val/nature "*******", 

    res = []
    sum = 0
    for img in os.listdir(tar_dir):
        img_tensor = processImg(os.path.join(tar_dir, img), model)
        res.append(img_tensor)
        # print(img)
        # sum += 1
        # if sum == 2:
        #     break

    res = torch.cat(res, dim=0)
    print(res.shape)
    # img_list = [processImg(os.path.join(tar_dir, img), model) for img in os.listdir(tar_dir)]

    return res

# "models/first_stage_models/vq-f4/config.yaml"
config_path = "configs/latent-diffusion/celebahq-ldm-vq-4.yaml" 
ckpt_path = "model.ckpt" # models/ldm/stable-diffusion-v1/  latent-diffusion

config = OmegaConf.load(config_path)
# model = instantiate_from_config(config.model) # autoencoder
with torch.no_grad():
    model = instantiate_from_config(config.model) 
    model.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"], strict=False)
    model = model.to("cuda").eval()

# sub_dir = 'ADM' + '/imagenet_ai_0508_adm/'
# sub_dir = 'BigGAN' + '/imagenet_ai_0419_biggan/'
# sub_dir = 'Midjourney/imagenet_midjourney/'
sub_dir = 'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/'
# sub_dir = 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/'
# sub_dir = 'VQDM/imagenet_ai_0419_vqdm/'
# sub_dir = 'glide/imagenet_glide/'
# sub_dir = 'wukong/imagenet_ai_0424_wukong/'
add_dir = sub_dir
nature_data = get_features(model, add_dir, "nature").cpu().numpy()
ai_data = get_features(model, add_dir, "ai").cpu().numpy()

nature_attack_column = np.zeros((nature_data.shape[0], 1))
ai_attack_column = np.ones((ai_data.shape[0], 1))
attack_column = np.vstack((nature_attack_column, ai_attack_column))
test_data = np.concatenate((nature_data, ai_data), axis=0)
test_data = np.hstack((test_data, attack_column))
print(test_data.shape)

train_time_stack = np.arange(nature_data.shape[0])
test_time_stack = np.arange(test_data.shape[0])

list_content = [str(i) for i in range(1, nature_data.shape[1] + 1)]
output_dir = 'data/' + sub_dir
os.makedirs(output_dir, exist_ok=True)
list_dir = output_dir + '/list.txt'
with open(list_dir, 'w') as f:
    for item in list_content:
        f.write(f"{item}\n")

train_columns = ['timestack'] + list_content
test_columns = train_columns + ['attack']
test_with_timestamp = np.column_stack((test_time_stack, test_data))
test_df = pd.DataFrame(test_with_timestamp, columns=test_columns)
test_file = os.path.join(output_dir, 'test_all.csv')
test_df.to_csv(test_file, index=False)

# labels_df = test_df.iloc[:, [0, -1]]
# test_df = test_df.drop(test_df.columns[[0, -1]], axis=1)
# labels_file = os.path.join(output_dir, 'test_label.csv')
# labels_df.to_csv(labels_file, index=False)

print("数据已保存到 csv 文件中。")

