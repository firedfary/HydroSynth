import torch
import math
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def initialize(module):
    torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    torch.nn.init.zeros_(module.bias)




class TimeEmbedding(torch.nn.Module):
    def __init__(self, T, d_model, dim) -> torch.Tensor:
        assert d_model % 2 == 0, "d_model不是2的倍数"
        super().__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)#类似transformer的位置编码器， 创建一个长度为d_model/2的向量，使用指数和对数缩放

        #时间步pos，从0到T-1
        pos = torch.arange(0, T).float()

        #时间步与emb相乘，得到时间嵌入矩阵，形状为T, d_model//2
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model//2]

        #对每个元素应用sin和cos，重塑为T, d_model
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model//2, 2]
        emb = emb.view(T, d_model)

        #嵌入层，MLP（线性，swish，线性）
        self.time_mebeding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(emb),
            torch.nn.Linear(d_model, dim),
            Swish(),
            torch.nn.Linear(dim, dim)
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                #使用xavier对网络中的权重进行初始化
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                torch.nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_mebeding(t)
        return emb
















#attention的能力和卷积的效率
#1.组归一化
#2.使用1*1的卷积生成QKV矩阵，1*1卷积抱持空间尺寸不变，同时运行跨通道的信息混合
#3.计算QK的注意力权重，应用到V上
#4.进行残差链接返回结果，允许网络绕过注意力模块
class AttenBlock(torch.nn.Module):
    def __init__(self, in_ch) -> None:#整个attendblock的参数只有一个，输入通道数
        super().__init__()
        #使用32个组进行组归一化
        self.group_norm = torch.nn.InstanceNorm2d(in_ch)
        #使用1*1的卷积生成QKV矩阵，1*1卷积抱持空间尺寸不变，同时运行跨通道的信息混合
        self.proj_q = torch.nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = torch.nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = torch.nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        #1*1的矩阵投影，进行进一步的特征转换
        self.proj = torch.nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            initialize(module=module)
        torch.nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        #应用组归一化生成qkv
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        #重塑qk计算注意力权重w
        q = q.permute(0, 2, 3, 1).view(B, H*W, C)
        k = k.view(B, C, H*W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H*W, H*W]
        w = torch.nn.functional.softmax(w, dim=-1)

        #重塑v，将权重w应用到v上
        v = v.permute(0, 2, 3, 1).view( B, H*W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H*W, C]
        #重塑结果回原始维度
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        #最后的1*1
        h = self.proj(h)

        return x + h



















class DownSample(torch.nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()

        #3*3卷积，步长为2
        self.main = torch.nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        initialize(self.main)

    def forward(self, x):
        x = self.main(x)
        return x

class UpSample(torch.nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        #3*3卷积，步长为1，wh不变
        self.main = torch.nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        initialize(self.main)

    def forward(self, x, tembed):#最后一个参数emb是啥,...不知道是啥，但缺了就会报错
        _, _, H, W = x.shape
        #插值放大
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x












class CrossAttention(torch.nn.Module):
    """条件交叉注意力层"""
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.to_q = torch.nn.Conv2d(query_dim, hidden_dim, 1, bias=False)
        self.to_k = torch.nn.Conv2d(context_dim, hidden_dim, 1, bias=False)
        self.to_v = torch.nn.Conv2d(context_dim, hidden_dim, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, query_dim, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        _, _, h_context, w_context = context.shape
        q = self.to_q(x).view(b, self.heads, -1, h*w)  # [B, heads, C_per_head, N_x]
        k = self.to_k(context).view(b, self.heads, -1, h_context*w_context)  # [B, heads, C_per_head, N_context]
        v = self.to_v(context).view(b, self.heads, -1, h_context*w_context)  # [B, heads, C_per_head, N_context]
        
        sim = torch.einsum('b h c i, b h c j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)












#残差模块
class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, atten=False, cond_dim=None):
        super().__init__()
        
        # ---------------------------
        # 第一部分：自适应组归一化+卷积
        # ---------------------------
        groups = min(8, in_ch)
        self.norm1 = torch.nn.GroupNorm(groups, in_ch)  # 替换InstanceNorm为GroupNorm
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        
        # AdaGN时间注入：将时间嵌入映射为scale和shift参数
        self.temb_proj = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(tdim, out_ch*2)  # 同时生成scale和shift
        )
        
        # ---------------------------
        # 第二部分：条件交叉注意力
        # ---------------------------
        self.atten = None
        if atten and cond_dim is not None:
            self.atten = CrossAttention(
                query_dim=out_ch,     # 主特征通道数
                context_dim=cond_dim, # 条件特征通道数（需与cond的通道数匹配）
                heads=4,
                dim_head=32
            )
        
        # ---------------------------
        # 第三部分：第二层卷积
        # ---------------------------
        self.norm2 = torch.nn.GroupNorm(8, out_ch)
        self.dropout = torch.nn.Dropout2d(dropout)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        # ---------------------------
        # 残差连接
        # ---------------------------
        if in_ch != out_ch:
            self.shortcut = torch.nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = torch.nn.Identity()
        
        # ---------------------------
        # 初始化
        # ---------------------------
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb, cond=None):
        # ---------------------------
        # 第一部分：AdaGN时间注入
        # ---------------------------
        h = self.norm1(x)
        
        # 从时间嵌入生成scale和shift (B, 2*out_ch)
        temb_params = self.temb_proj(temb)
        scale, shift = temb_params.chunk(2, dim=1)  # 各(B, out_ch)
        
        # 应用自适应归一化
        h = self.conv1(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        
        # ---------------------------
        # 第二部分：条件交叉注意力
        # ---------------------------
        if self.atten is not None and cond is not None:
            # 假设cond已经通过下采样与当前分辨率对齐
            h = self.atten(h, cond)  # [B, out_ch, H, W]4,128,32,32;4,64,2,2
        
        # ---------------------------
        # 第三部分：第二层卷积
        # ---------------------------
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.shortcut(x)
    







