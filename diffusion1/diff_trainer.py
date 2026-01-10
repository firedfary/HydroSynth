import torch
import numpy as np
# %config InlineBackend.figure_format = 'svg' 这个只能在jupyter notebook中使用
import diffusers
from fucs import save_image
import random

def extract(v, t, x_shape):
    #提取系数作为 timestep， reshape成为[batch_size, 1, 1, .....]的形式（为了方便广播）
    device = t.device
    out = torch.gather(v.to(device), index=t, dim=0).float()
    return out.view([t.shape[0]] + [1]*(len(x_shape)-1))


class GaussianDiffusionTrainer(torch.nn.Module):
    def __init__(self, mode1, bata1, bataT, T):
        super().__init__()
        self.model = mode1
        self.T = T
        
        # batas = torch.linspace(bata1, bataT, T)
        # self.register_buffer('batas', batas)#是模型的参数，但不会改变
        # alphas = 1-self.batas#这里batas不亮可能是self.batas 没有定义
        # alphas_bar = torch.cumprod(alphas, dim=0)#计算alpha的累计乘积
        # alphas_bar_prev = torch.nn.functional.pad(alphas_bar, [1,0], value=1)[:T]

        # self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        # self.register_buffer('sqrt_1_alpha_bar', torch.sqrt(1. - alphas_bar))
        # self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        # self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        # self.register_buffer('posterior_var', self.batas * (1. - alphas_bar_prev)/(1. - alphas_bar))
        self.ddim_sch = diffusers.schedulers.scheduling_ddim.DDIMScheduler(num_train_timesteps=T, beta_schedule='squaredcos_cap_v2',)

    # def predict_xt_prev_mean_from_eps(self, xt, t, eps):#eps就是noise predicter的预测结果
    #     assert xt.shape == eps.shape
    #     return extract(self.coeff1, t, xt.shape) * xt - extract(self.coeff2, t, xt.shape) * eps

    def forward(self, x0, conditions, mask, train_mode='eps'):
        t = torch.randint(int(self.T), size=(x0.shape[0],), device=x0.device)
        # torch.manual_seed(42)
        noise = torch.randn_like(x0.float(), device=x0.device)
        x0 = torch.nan_to_num(x0, nan=0.0)
        xt = self.ddim_sch.add_noise(x0, noise, t)
        #xt是加噪的图像，t是时间步，noise是噪声
        # encoder_hidden_states = conditions.flatten(2).permute(0, 2, 1)


        model_out = self.model(xt, t, conditions)
        # save_image(x0, xt, noise, model_out, names=['x0', 'xt', 'noise', 'eps'], show=True)
        # model_out = self.model(xt, t, conditions)

        print(f"t.cpu:{t.cpu()}")
        if train_mode == 'eps':
            target = noise
        elif train_mode == 'x0':
            target = x0
        else:
            raise NotImplementedError()
        mse = torch.nn.functional.mse_loss(model_out, target, reduction='none')
        # loss = (mse * mask).sum() / mask.sum()
        
        return mse













class GaussFiffusionSampler(torch.nn.Module):
    def __init__(self, model, bata1, bataT, T) -> None:
        super().__init__()

        self.model = model
        self.T = T

        batas = torch.linspace(bata1, bataT, T).double()
        self.register_buffer('batas', batas)
        alphas = 1-self.batas#这里batas不亮可能是self.batas 没有定义
        alphas_bar = torch.cumprod(alphas, dim=0)#计算alpha的累计乘积
        alphas_bar_prev = torch.nn.functional.pad(alphas_bar, [1,0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.batas * (1. - alphas_bar_prev)/(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, xt, t, eps):#eps就是noise predicter的预测结果
        assert xt.shape == eps.shape
        return extract(self.coeff1, t, xt.shape) * xt - extract(self.coeff2, t, xt.shape) * eps
    
    def P_mean_variance(self, xt, condition, t):
        var = torch.cat([self.posterior_var[1:2], self.batas])
        var = extract(var, t, xt.shape)

        eps = self.model(torch.cat((xt, condition), dim=1), t)#eps时预测结果噪声，下一步才是减去噪声
        if float(t[0]) % 100 == 0:
            save_image(eps, names=[str(int(t[0]))+'eps'], path='E:/D1/diffusion/my_models/my_model_data/picture/')
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(xt=xt, t=t, eps=eps)
        return xt_prev_mean, var#减去噪声的结果
    
    #逆向的过程
    def forward(self, x_T, condition, save_path):
        x_t = x_T
        save_image(x_t, names=[str(self.T)], path=save_path)
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step

            mean, var = self.P_mean_variance(x_t, condition, t)
            # if time_step > 0:
            #     noise = torch.rand_like(x_t.float())
            # else:
            #     noise = 0
            
            x_t = mean
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor"

            
            if (self.T - time_step) % 100 == 0 or time_step ==0:
                save_image(x_t, names=[str(time_step)], path=save_path)
                
        x_0 = x_t
        return x_0







