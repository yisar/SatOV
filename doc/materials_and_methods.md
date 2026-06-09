# 材料与方法

## 模型概览
本工作基于 CLIP / OpenCLIP 视觉-文本模型，提出一种稠密（pixel-level）语义特征提取与零样本分割流程。核心模块包括：视觉特征提取（保留 Transformer 前若干层并对最后一层采用 Self-Self Attention）、去偏置（Debiasing）、引导上采样（UPA 或 AnyUp），以及基于文本模板的零样本分类器初始化。

## 1. 特征提取与位置编码插值
输入图像为 $x\in\mathbb{R}^{B\times 3\times H\times W}$，首先经过 CLIP 的第一层卷积得到低分辨率特征：

$$
X_{in}=\text{conv1}(x),\quad X_{in}\in\mathbb{R}^{B\times D\times H'\times W'}
$$

将空间维展开并拼接 CLS token 得到 token 序列：

$$
X_{tokens}=\text{concat}(\text{CLS},\text{flatten}(X_{in}))\in\mathbb{R}^{B\times N\times D}
$$

其中位置编码 $P$ 若与当前网格大小不匹配，则通过双三次插值进行重采样，保证 token 与位置编码一一对应。

（解释）: 通过插值对位置编码做适配，可以让预训练的 ViT 模型在不同输入分辨率下仍然保持合理的相对位置感知，从而减少重训练成本。

## 2. Self-Self Attention（最后一层）
我们在保留前 L-1 层 Transformer 前馈之后，对最后一层采用一种只依赖 Query 的自内积注意力（代码中称为 Self-Self Attention）：

首先计算线性变换得到 Q、K、V：

$$
[Q,K,V]=X_{norm}W_{in}+b_{in}
$$

代码中将 Attention 的 K 用 Q 替代，所以注意力矩阵为：

$$
A=\text{softmax}\left(\frac{QQ^{\top}}{\sqrt{d_h}}\right)\in\mathbb{R}^{B\times H_{head}\times N\times N}
$$

输出为：

$$
\text{AttnOut}=A\,V,\quad X_{feat}=\text{AttnOut}W_{out}+b_{out}
$$

（解释）: 该 Self-Self 设计强调 patch 间的相互关系而不依赖全局的 K，从而突出局部上下文一致性，适用于稠密预测场景如语义分割或像素级检索。

## 3. 去偏置（Debiasing）
在得到最后的 token 特征后，分离 CLS token 与 Patch tokens，并对每个 patch 做去偏置操作：

$$
F_{patch}=F[:,1:, :] - F[:,0:1, :]
$$

（解释）: CLS token 捕获图像的全局语义或背景偏置信息，通过从每个 patch 中减去对应的 CLS 表征，可降低背景干扰、增强局部语义对比度，从而提升像素级分类的精确度。

## 4. 引导上采样（Guided Upsampling）
去偏置得到的低分辨率特征 $F_{lr}\in\mathbb{R}^{B\times D\times H'\times W'}$ 需要上采样回原始图像分辨率。采用三种策略的优先级：UPA（若有自定义实现）、AnyUp（通过 torch.hub 加载）、或双线性插值：

$$
F_{hr}=\begin{cases}
\text{UPA}(guide, F_{lr}), & \text{if UPA available}\\
\text{AnyUp}(guide, F_{lr}), & \text{elif AnyUp available}\\
\text{Interp}(F_{lr}), & \text{else}
\end{cases}
$$

（解释）: 使用引导图（原始图像或高分辨率引导）可以在上采样时保留边界与细节信息，提升最终像素级预测的空间对齐性。

## 5. 视觉投影与归一化
上采样后通过 $1\times1$ 卷积将视觉通道映射到文本嵌入维度：

$$
F_{embed}=\text{Conv}_{1\times1}(F_{hr})\in\mathbb{R}^{B\times C\times H\times W}
$$
对通道维进行 L2 归一化：

$$
\hat{F}_{p}=\frac{F_{embed,p}}{\|F_{embed,p}\|_2}
$$

其中 $p$ 表示像素位置。

## 6. 零样本分类器初始化（Zero-shot）
对于每个类别组 $i$（允许包含近义词集合 $S_i$），以及一组文本模板 $T$，先将模板化文本 token 化并通过 CLIP 文本编码器得到向量，然后对模板与近义词取均值并归一化，构成类别权重：

$$
\bar{e}_i=\frac{1}{|S_i||T|}\sum_{s\in S_i}\sum_{t\in T}\text{normalize}\big(\text{encode}(t(s))\big)
$$
$$
w_i=\frac{\bar{e}_i}{\|\bar{e}_i\|_2}
$$

最终将所有类别权重拼成矩阵 $W=[w_1,\dots,w_C]\in\mathbb{R}^{D\times C}$（代码中为 `zeroshot_weights`）。

（解释）: 通过对近义词与模板求均值，可以获得对类别概念更鲁棒的表示，减少单一描述带来的偏差。

## 7. 前向推断与相似度计算
对归一化后的像素特征 $\hat{F}_p$ 与类别权重 $w_i$ 做内积并除以温度参数 $\tau$（代码中为 `temperature`），得到每个像素对每个类别的 logits：

$$
\text{logits}_{p,i}=\frac{\hat{F}_p^{\top} w_i}{\tau}
$$

在实现上，先将特征重排为矩阵形式做一次批量矩阵乘法，然后恢复为 $[B, C, H, W]$ 的形状输出。

## 实验设置（可选）
- 设备：优先使用 GPU（若可用），否则使用 CPU。
- 预训练模型：通过 `open_clip.create_model_and_transforms` 加载指定的 CLIP / OpenCLIP 权重。
- 温度初始化：代码中将温度设为可学习标量，初始值为 $\tau=0.07$。


## 小结
本文的材料与方法关键点在于：
- 在 CLIP 视觉分支中保留大部分预训练结构，同时对最后一层采用 Self-Self Attention 强化 patch 间局部关系；
- 通过去偏置操作减弱全局背景干扰；
- 使用引导上采样保持高分辨率细节；
- 使用模板与近义词集合构造稳健的零样本类别权重，结合像素级相似度实现零样本分割/分类。

如需我将该文本合并到已有论文模板中，或生成英文版与参考文献，请告诉我。