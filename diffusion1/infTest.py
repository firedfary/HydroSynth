import torch
import Unetwork
import os
import numpy as np
import HydroSynth.diffusion.diff_trainer as diff_trainer
import coders
import matplotlib.pyplot as plt
from fucs import save_image
import shutil
import config

class ref_dataset(torch.utils.data.Dataset):
    def __init__(self, condition, target):
        self.condition = condition
        print("valid file number:", str(len(self.condition)))
        print("shape of training:", self.condition.shape)
        self.data = torch.randn(self.condition.shape[0], 1, 64, 64)
        self.target = target
    def __getitem__(self, index):
        return self.data[index], self.condition[index], self.target[index]

    def __len__(self):
        return len(self.data)    # self.data是一个变量
    
    
def eval(modelconfig: dict):
    with torch.no_grad():
        device = torch.device(modelconfig['device'])
        model = Unetwork.UNet(
            T=modelconfig["T"],
            in_ch=5,
            ch=modelconfig["channel"],
            ch_mult=modelconfig["channel_mult"],
            atten=modelconfig["atten"],
            num_res_block=modelconfig["num_res_block"],
            dropout=modelconfig["dropout"]
        )
        ckpt = torch.load(
            os.path.join(modelconfig['save_weight_dir'], modelconfig['eval_load_weight']),
            map_location=device
        )
        model.load_state_dict(ckpt)
        model.eval()

        sampler = diff_trainer.GaussFiffusionSampler(
            model=model,
            bata1=modelconfig['bata_1'],
            bataT=modelconfig['bata_T'],
            T=modelconfig['T']
        ).to(device)


        target_data_list = []
        target_file_dir = modelconfig["hr_path"]
        target_file_list = os.listdir(target_file_dir)
        for file in target_file_list:
            file_path = os.path.join(target_file_dir, file)
            data = np.load(file_path)#36,64,64
            target_data_list.append(data)

        target_data = torch.from_numpy(np.stack(target_data_list).astype(np.float32))#340,36,64,64
        data_mask = ~torch.isnan(target_data)
        target_data = torch.unsqueeze(target_data, dim=2).view(-1, 1, 64, 64)#12240,1,64,64
        # np.save('E:/D1/diffusion/my_models/my_model_data/picture/real4.npy', target_data.squeeze().cpu().numpy())
        data_mask = torch.unsqueeze(data_mask, dim=2).view(-1, 1, 64, 64)
        num_patch = len(target_data_list)


        sst = torch.from_numpy(np.load(modelconfig["sst_path"]).astype(np.float32))#36,3,64,64
        sst = torch.unsqueeze(sst, dim=0).repeat(num_patch, 1, 1, 1, 1).view(-1, 3, 64, 64)#12240,3,64,64
        model_file_dir = modelconfig["lr_path"]
        model_file_list = os.listdir(model_file_dir)
        model_data_list = []
        for file in model_file_list:
            file_path = os.path.join(model_file_dir, file)
            data = np.load(file_path)#276,64,64
            model_data_list.append(data)
        model_data = torch.from_numpy(np.stack(model_data_list).astype(np.float32))#340,36,64,64
        model_data = torch.unsqueeze(model_data, dim=2).view(-1, 1, 64, 64)#12240,1,64,64
        condition = torch.cat((sst, model_data), dim=1)#12240,4,64,64

        dataset = ref_dataset(condition, target_data)
        # data, condition = dataset[:]
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=modelconfig["batch_size"],
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

        # all_x_0_list = []
        # for i, (data, condition, target) in enumerate(dataloader):
        #     data = data.to(device)
        #     condition = condition.to(device)
        #     target = target.to(device)
        #     if torch.any(torch.isnan(target)):
        #         print("target has NaN")
        #         continue

        #     x_0 = sampler(data, condition, modelconfig['picture_save_path'])
        #     ACC = coders.cal_acc(x_0.squeeze(), target.squeeze())
        #     print(ACC)
        #     # x_0 = torch.randn(condition.shape[0], 1, 64, 64).to(device)
        #     all_x_0_list.append(x_0.cpu())
        # all_x_0 = torch.cat(all_x_0_list, dim=0)

        
        for i in range(12240):
            data, condition, target = dataset[i:i+32]
            if torch.any(torch.isnan(target)):
                print("target has NaN")
                continue
            save_image(data, condition, target, names=['data', 'condition', 'target'], path=modelconfig['picture_save_path'])
            data = data.to(device)
            condition = condition.to(device)
            target = target.to(device)
            print(target)
            x_0 = sampler(data, condition, modelconfig['picture_save_path'])
            save_image(x_0, names=['eval'], path=modelconfig['picture_save_path'])
            ACC = coders.cal_acc(x_0.squeeze(), target.squeeze())
            np.save(modelconfig["picture_save_path"]+'eval_x02.npy', x_0.cpu().numpy())
            np.save(modelconfig["picture_save_path"]+'real_target2.npy', target.cpu().numpy())
            np.save(modelconfig["picture_save_path"]+'data_noise2.npy', data.cpu().numpy())
            np.save(modelconfig["picture_save_path"]+'condition2.npy', condition.cpu().numpy())
            print(ACC)
            break

        # ACC = coders.cal_acc(all_x_0.squeeze(), target_data.squeeze())
        # print(torch.isnan(all_x_0).any())
        # print(torch.isnan(target_data).any())#目标数据有NaN
        # np.save('E:/D1/diffusion/my_models/my_model_data/picture/eval2.npy', all_x_0.squeeze().numpy())
        # np.save('E:/D1/diffusion/my_models/my_model_data/picture/real2.npy', target_data.squeeze().numpy())
        # # diff_trainer.save_image(data, modelconfig['picture_save_path'], -1)
        # print(ACC)
        # plt.plot(ACC.detach().numpy())
        # plt.show()

        # fig, axs = plt.subplots(1, 2, figsize=(7, 6))
        # im1 = axs[0].imshow(torch.squeeze(x_0[0]).detach().cpu().numpy())
        # cbar = fig.colorbar(im1, ax=axs[0])
        # axs[0].set_title('recons Image')
        # axs[0].axis('off')
        # im2 = axs[1].imshow(torch.squeeze(data[0]).detach().cpu().numpy())
        # cbar = fig.colorbar(im2, ax=axs[1])
        # axs[1].set_title('Original Image')
        # axs[1].axis('off')
        # plt.show()


if __name__ == '__main__':
    modelconfig = config.modelconfig
    shutil.rmtree(modelconfig['picture_save_path'], ignore_errors=True)  # 删除文件夹及其内容
    os.makedirs(modelconfig['picture_save_path'])  # 重新创建空文件夹

    eval(modelconfig)