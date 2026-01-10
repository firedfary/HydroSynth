import torch
import UnetModels
'''
先把条件转成n*64*64的区块，预测量跟条件的关系由
每一个预测量区块跟条件的关系不一样
训练的时候，对每一个区块对条件都是平等的。
每一个模式值对应一个观测值，在模式值里面加入条件
'''
class condition_conv(torch.nn.Module):
    def __init__(self, conditions) -> None:
        super().__init__()
        #INPUT（输入层）-CONV（卷积层）-RELU（激活函数）-POOL（池化层）-FC（全连接层）
        self.condition = conditions#276,1,20,20
        self.min_len = min(self.condition.shape[-1], self.condition.shape[-2])
        if self.min_len < 64:
            self.scale_factor = (64//self.min_len)+1
            self.condition = torch.nn.functional.interpolate(self.condition, scale_factor=self.scale_factor, mode='bicubic')
        self.high = self.condition.shape[-2]#(276,1,80,240)a, b
        self.wide = self.condition.shape[-1]#240
        self.kernel_size1 = (self.high//64,self.wide//64)
        self.kernel_size2 = (int((self.high-4)/ self.kernel_size1[0]-64+1), int((self.wide-4)/ self.kernel_size1[1]-64+1))

        self.conv_blocks = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(1),
            torch.nn.Conv2d(1, 2, kernel_size=5), #a-4, b-4;76,236
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=self.kernel_size1),
            torch.nn.Conv2d(2, 3, kernel_size=self.kernel_size2),
            #torch.nn.ELU(),
        )


    def forward(self, conditions):
        return self.conv_blocks(conditions)


