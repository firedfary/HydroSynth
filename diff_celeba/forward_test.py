import HydroSynth.diffusion.diff_trainer as diff_trainer
import torch
import os
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bata1 = 0.001
bataT = 0.01

def model(xt, t):
    return torch.rand_like(xt)

a_trainer = diff_trainer.GaussianDiffusionTrainer(mode1=model, bata1=bata1, bataT=bataT, T=2000)

# x0 = torch.randint(1,255,(1,512,512), dtype=torch.float).to(device='cpu')
# x0 = torch.full((1, 255, 255), 150.)


folder_path = 'my_model/x0s'  # 修改为您自己的文件夹路径

# 遍历文件夹下所有图片文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg', '.jpeg', '.png')):
#         # 读取图片
#         image_path = os.path.join(folder_path, filename)
#         image = cv2.imread(image_path)

#         # 裁剪为512x512大小
#         if image is not None:
#             image = image[:512, :512]  # 裁剪左上角512x512大小的区域
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             tensor = torch.from_numpy(image)
#             x0 = tensor.permute(2, 0, 1)
#             break



image = cv2.imread('my_model/x0s/5.jpg')
step = 500

# 裁剪为512x512大小
if image is not None:
    image = cv2.resize(image, (512,512))  # 裁剪左上角512x512大小的区域
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(image)
    x0 = tensor.permute(2, 0, 1)



print(x0.shape[0])
a_trainer.forward(x0, 0)
