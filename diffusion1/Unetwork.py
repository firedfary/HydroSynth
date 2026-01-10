import torch
import UnetModels


class UNet(torch.nn.Module):#注意下面的那个ch不是尺寸而是通道数，真的有128个通道，由3通道卷积而来
    #T时间步数，ch尺寸，ch_mult=[1,2,2,2]通道数乘列表，num_res_block残差块的数量，atten是一个长度为1，值为1的列表
    def __init__(self, T, in_ch, ch, ch_mult, atten, num_res_block, dropout, cond_dim=10) -> None:
        super().__init__()
        assert all([i < len(ch_mult) for i in atten]), '需要开启注意力机制的模板编号不能大于通道数乘列表的长度'
        self.cond_net = torch.nn.Sequential(
        torch.nn.Conv2d(cond_dim, cond_dim, 3, padding=1),
        *[UnetModels.DownSample(cond_dim) for _ in ch_mult]
        )


        #时间嵌入的通道数 ch(128) * 4(通道) = 512
        tdim = ch * 4
        #时间嵌入
        self.time_embedding = UnetModels.TimeEmbedding(T=T, d_model=ch, dim=tdim)


        #那128个通道就是这里来的
        self.head = torch.nn.Conv2d(in_channels=in_ch, out_channels=ch, kernel_size=3, stride=1, padding=1)

        #下采样
        self.downblocks = torch.nn.ModuleList()
        chs = [ch]
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult#通道数128*通道乘数列表  列表里一共四个值，代表下采样四次
            for _ in range(num_res_block):                                                 #atten中第i个值在atten中，就有注意力机制
                self.downblocks.append(UnetModels.ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, atten=(i in atten), cond_dim=cond_dim))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult)-1:
                self.downblocks.append(UnetModels.DownSample(now_ch))
                chs.append(now_ch)

        self.temb_layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(tdim, tdim), torch.nn.SiLU()) for _ in range(len(self.downblocks))])
        

        #中间块，两个resblock，一共有atten
        self.middleblocks = torch.nn.ModuleList([
            UnetModels.ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, atten=True),
            UnetModels.ResBlock(now_ch, now_ch, tdim=tdim, dropout=dropout, atten=False)
        ])

        #上采样
        self.upblocks = torch.nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_block + 1):
                self.upblocks.append(UnetModels.ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, atten=(i in atten)
                ))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UnetModels.UpSample(now_ch))

        assert len(chs) == 0, 'chs不等于零,还每采样完'
        self.tail = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(now_ch),
            UnetModels.Swish(),
            torch.nn.Conv2d(in_channels=now_ch, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.initialize()


    def initialize(self):
        torch.nn.init.kaiming_normal_(self.head.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.head.bias)
        torch.nn.init.zeros_(self.tail[-1].weight)
        torch.nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, cond):
        
        #时间嵌入
        tembed = self.time_embedding(t)
        temb_multi = [layer(tembed) for layer in self.temb_layers]
        h = self.head(torch.cat([x, cond], dim=1))
        hs = [h]

        #U-Net
        j = 1
        for i, layer in enumerate(self.downblocks):
            if isinstance(layer, UnetModels.ResBlock):
                cond_features = self.cond_net[0:j](cond)
                h = layer(h, temb_multi[i], cond=cond_features)  # 传递cond到每个ResBlock 4,64,32,32; 4,256; 4,10,32,32
            else:
                j += 1
                h = layer(h)
            hs.append(h)
        
        for layer in self.middleblocks:
            h = layer(h, tembed)

        for layer in self.upblocks:#71
            if isinstance(layer, UnetModels.ResBlock):#如果有resblock
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, tembed)
        
        h = self.tail(h)

        assert len(hs) == 0, 'hs不为零,还有模块和流程没走完'
        return h #例子说这里的hshape是[8, 3, 32, 32]









