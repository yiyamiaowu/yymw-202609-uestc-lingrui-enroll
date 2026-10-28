## 任务三：我有说明书，太好啦
### 环境搭建：小菜一碟
![环境截图](../doc/environment.png "环境截图")
### 实现一个简单的模型
1. 基本模型
```
0 I_rainy
1 Conv2d 3x3, 3→64
2 Residual Block x 8(
    input 
    Conv2d 3x3
    BatchNorm2d
    ReLU
    Conv2d 3x3
    BatchNorm2d
    ReLU  latest_result+ input
    output
)
3 Conv2d 3x3, 64→3
4 I_clean = I_rainy - R_hat
```
2. 完整代码
```
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,channels = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,
                stride=1, padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,
                stride=1, padding=1,bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ReLU = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.ReLU(self.body(x) + x)

class BaselineNet(nn.Module):
    def __init__(self, in_out_channel = 3, channels = 64, num_blocks = 8):
        super().__init__()
        self.head = nn.Conv2d(in_out_channel,channels,
            kernel_size=3,stride=1, padding=1,bias=False)
        self.block = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]   # *解包列表
        )
        self.tail = nn.Conv2d(channels,in_out_channel,
            kernel_size=3,stride=1, padding=1,bias=False)
    
    def forward(self, x):
        feat = self.head(x)
        feat = self.block(feat)
        rain = self.tail(feat)
        clean = x - rain
        return clean
```
#### 针对此问题，请你思考并寻找可能的解决方案
根据 CNN 的卷积计算原理，一个 3x3 的卷积核在当前层只能感受到附近 3x3 区域的信息，无法捕捉长距离依赖（例如贯穿图片的雨线）。在分类任务中，模型通过池化层不断减小特征图尺寸来扩大感受野，但在去雨任务中，我们通常希望保持原始分辨率。这意味着需要堆叠非常深的网络才能让像素间进行远距离信息交流，而深层网络会带来梯度消失、训练困难等问题。
ps:我自己想怎么可能想的来，最多就想到多尺度分支，注意力这些我在做前面问题时学到的一些方法
1. 多尺度分支：把同一张特征图送入不同分辨率（不下采样加上不同下采样）的分支并行处理，再融合
2. 空洞卷积：
将3*3卷积由
■ ■ ■ 
■ ■ ■ 
■ ■ ■ 
变成
■ □ □ ■ □ □ ■ 
□ □ □ □ □ □ □ 
□ □ □ □ □ □ □ 
■ □ □ ■ □ □ ■ 
□ □ □ □ □ □ □ 
□ □ □ □ □ □ □ 
■ □ □ ■ □ □ ■ 
在不改变参数量的情况下让卷积核覆盖更大的区域
但容易有一些地方覆盖不到
3. U-Net 编解码
    1. 编码器多次卷积+下采样
    2. 每次下采样产生的特征图直接与解码时的反卷积产生的特征图拼接
    3. 通道数翻倍
    4. 卷积变回原通道数
也就是与多尺度分支结合+跳跃连接
4. 注意力
    1. 通道注意力
    对特征图每个通道乘上一个权重s
    $$s=sigmoid(W_2ReLU(W_1z))，z=\sum_{i,j}p_{ij}$$
    p是特征图每个通道全局平均池化值，$W_1,W_2$是全连接层，s正向传播时会计算，反向传播时会更新参数
    2. 空间注意力
    生成一张与原图一样大小的权重图，告诉模型那些地方应该注意
5. 非局部模块
$$y_i=\frac{1}{C(x)}\sum_{j}f(x_i,x_j)g(x_j)$$
$C(x)$为归一化因子，$f(x_i,x_j)$为位置$x_j$对$x_i$的关系，$g(x_j)$为对$x_j$的变换
意义为根据$x_j$以及位置$x_j$对$x_i$的关系得到新特征$y_i$，再经过经过变换和残差连接加回原特征$x_i$
6. Transformer
ps:看不懂定义，尽量自己解释一下
一个窗口在特征图上滑动，对窗口内的进行注意力加权
7. 递归/渐进式网络
就和前面写的代码一样，重复执行一个模块ResidualBlock，进行多次学习雨层并反复优化
### 这个代码是什么意思？
1. 以下是运行结果
    !["运行结果"](..\\doc\\baseline_result.png "运行结果")
2. 之前再C语言里就学过可以通过命令行往main里面传参数，这下在python里也会了
    如何实现：
    1. 首先导入argparse库
    2. 然后最好是写一个用来获取命令行参数的函数
    3. 在函数中定义一个自己的`self_parser=argparse.ArgumentParser(description="程序说明")`
    4. 之后在函数中写`parser.add_argument(--name,type="")`下面是可填的参数

    |参数|	作用|
    |---|---|
    |--name|	参数名|	
    |type|	类型转换|
    |default|	默认值|
    |required|	是否必填|
    |choices|	限定可选值|
    |help|	帮助说明|
    |action|	特殊行为|
    |nargs|接受值数量|

    布尔值不能直接写，要使用`parser.add_argument("--name",action="store_true")`
    6. 调用该函数
3. 最初没写考虑mse为0的情况