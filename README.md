- [ ] 第一阶段：选题与开题（确定航向）

这是最关键的一步，好的选题等于成功了一半。

1. **确定领域**：从你感兴趣或成绩较好的课程中寻找灵感。
2. **缩小范围**：**切忌题目过大**（例如：不要写“论中国经济”，要写“短视频直播带货对XX市大学生消费观念的影响”）。题目越小，越容易写深、写透。
3. **文献搜索**：在知网（CNKI）、万方或 Google Scholar 上搜索关键词，确认该题目是否有足够的参考资料，且是否还有研究空间。
4. **撰写开题报告**：明确你的研究目的、研究方法（问卷、访谈、案例分析或实验）和技术路线。

- [ ] 第二阶段：文献综述与资料收集（储备弹药）

不要急着动笔写正文，先看别人是怎么说的。

- **泛读与精读**：先看摘要和结论，筛选出 10-20 篇核心文献精读，并做好笔记。
- **建立框架（提纲）**：根据文献阅读心得，拟定详细的三级大纲。
	- *引言*：为什么研究这个？
	- *现状/理论基础*：目前是什么情况？
	- *问题分析*：存在什么问题？
	- *对策建议*：怎么解决？
	- *结论*：总结全文。

- [ ] 第三阶段：初稿撰写（攻坚克难）

这个阶段的目标是**“先完成，再完美”**。

1. **数据/案例填充**：如果你有调研数据或实验结果，先完成这部分核心章节。
2. **模块化写作**：不一定要从第一章开始。可以先写你最擅长的部分，最后写引言和摘要。
3. **规范引用**：写的时候随手标记引用来源，否则后期回过头去找参考文献会非常痛苦。

- [ ] 第四阶段：格式调整与查重（精雕细琢）

本科论文的老师有时更看重你的态度（格式）而非学术高度。

- **格式检查**：严格按照学校下发的“论文格式模版”调整目录、字体、行间距、注释和参考文献格式。
- **预查重**：在提交学校前，使用正规渠道进行初次查重，根据查重报告修改重复率较高的段落（采用**“删、改、换”**策略，即删除冗余、改写句式、更换词汇）。

- [ ] 第五阶段：答辩准备（最终冲刺）

- **制作 PPT**：逻辑清晰，突出你的研究重点和结论，不要放太多文字。
- **模拟自测**：预测导师可能会问的问题（如：你的样本量为什么这么小？你的核心创新点在哪？）。



**论文**

《项远哲-毕业论文》《TR-EdgeLD》《面向边缘智能的边端协作卷积神经网络推理加速研究_周金粮》

《面向边缘智能的神经网络计算方法研究_薛峰》

**大纲**

1、我需要完成本科毕业设计的论文，所以现在要求你生成我毕设论文的大纲

2、我的毕设的创新点是在论文《TR-EdgeLD》的思路上进行改进的，我通过对VGG每一层的传输、预处理、卷积时间的统计，发现部分时间开销存在优化空间，然后在 层与层之间输入输出块的分割 以及 子节点与主节点进行通信的空间位置 上进行优化而得出的

3、《面向边缘智能的神经网络计算方法研究_薛峰》是《TR-EdgeLD》论文一作作者的硕士毕业论文，其中包含对论文《TR-EdgeLD》的阐述；《面向边缘智能的边端协作卷积神经网络推理加速研究_周金粮》是与我方向相关的毕业论文；《项远哲-毕业论文》是我们学校以往学长的毕业论文

4、我希望你依照《项远哲-毕业论文》的大纲结构，结合《面向边缘智能的边端协作卷积神经网络推理加速研究_周金粮》

《面向边缘智能的神经网络计算方法研究_薛峰》大纲内容以及《TR-EdgeLD》的论文内容，生成我的毕设论文大纲

**第一版代码**

1、Edge_LD是主项目代码，AdapCP是需要借鉴的项目代码，pdf文件是AdapCP项目的论文，md文件是AdapCP应用到Edge_LD的可行性分析

2、我需要你提取AdapCP的创新思路，应用在Edge_LD项目上，需要有warm_up环节

我希望你将AdapCP的关于跨层划分（Inter-layer Partition）、层内结构划分（Intra-layer Partition）以及基于DRL的动态划分策略的思路应用到EdgeLD中（其他不作考虑），并完成具体的代码实现



cd d:\Graduate\WorkBench\work_1\Edge_LD\node_test

模型只使用VGG13以及16，其他模型不作考虑

不需要考虑第五步的窃取风险分析



1、阅读"D:\Graduate\WorkBench\work_1\"文件夹下的AdapCP3765961.pdf，这个是当前路径AdapCP文件夹项目代码的论文

2、主要分析论文中的第四五章，我需要把这个思路应用在当前路径Edge_LD文件夹项目上



AdapCP

1、背景，简介，结构

2、主要创新点的背景，创新的点是什么

3、对层间算法ILP和层内算法的介绍DRL

4、对层间算法ILP的可行性分析

5、对层内算法DRL的可行性分析（深度强化学习，深度确定性策略梯度DDPG）

6、自适应，动态环境处理（不作考虑

7、预防模型窃取（不作考虑

8、实验设备和实验结果

9、结论和未来工作

**理解核心**：层间 ILP 算法，层内 DRL 算法

ILP

（1）哪几层给哪个节点执行...

DRL

（1）拆分通道而非宽度...



为什么其他方法的优化效果并没有这么明显

1、我是专门针对VGG的模型进行优化的，而其他基本都会考虑泛用性（基本都只会考虑一层一层的执行），因此时间的大头（VGG个别层面的传输成本）并没有得到有效的解决

2、我这只能在静态环境才能达到如此的效果，放在动态环境中那就不一定了

我突然发觉，直接按照通道划分





1、读取全文内容，这个是当前路径AdapCP文件夹项目代码的论文

2、主要分析论文中的第四五章，我需要把这个思路应用在当前路径Edge_LD文件夹项目上



CIFAR-10



❗Traceback (most recent call last):
  File "namenode_0_4_warm.py", line 374, in <module>
    run_adapcp_inference(namenode, adapcp_namenode, round_idx)
  File "namenode_0_4_warm.py", line 266, in run_adapcp_inference
    epsilon=0.0
  File "namenode_0_4_warm.py", line 103, in compute_ddpg_ratios
    computing_power_list, network_bw_list, current_load, epsilon
  File "..\node_test\adaptive_partitioner.py", line 41, in get_partition_ratios
    self.current_action = self.agent.select_action(self.current_state, epsilon=epsilon)
  File "..\node_test\ddpg_agent.py", line 191, in select_action
    probs = self.actor(state_tensor).detach().numpy()[0]
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\modules\module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "..\node_test\ddpg_agent.py", line 173, in forward
    return self.net(state)
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\modules\module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\modules\container.py", line 92, in forward
    input = module(input)
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\modules\module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\modules\linear.py", line 87, in forward
    return F.linear(input, self.weight, self.bias)
  File "E:\condaData\envs_dirs\EdgeLD\lib\site-packages\torch\nn\functional.py", line 1370, in linear
    ret = torch.addmm(bias, input, weight.t())
RuntimeError: size mismatch, m1: [1 x 12], m2: [9 x 256] at C:\w\1\s\windows\pytorch\aten\src\TH/generic/THTensorMath.cpp:197

🛠️state_dim = 12

❗

(E:\condaData\envs_dirs\EdgeLD) d:\Graduate\WorkBench\work_1\Edge_LD\node_test>python datanode_0_4_warm.py

 DataNode 0 持久化启动（AdapCP Filter Splitting模式） 
DataNode 0 开始初始化
Hello DataNode 0, I'm NameNode
DataNode 0 初始化完成

----- DataNode 0 第 1 轮推理开始 -----
DataNode recv data length:  702491
DataNode recv data length:  702491
接收来自 NameNode 的数据 recv_tensor: torch.Size([1, 14, 112, 112])
要求计算 4 - 18
datanode.get_last_inference_layer(): 0
第 1 轮发生错误: 'NoneType' object has no attribute 'size'
关闭 DataNode 0 的Socket连接
DataNode0 持久化连接已关闭





---



### ⚠️ 需要核心纠正的地方（论文思路与你代码的偏差）

在你目前的 `ilp_solver.py` 和 `adaptive_partitioner.py` 中，存在两个与论文架构设计不匹配的地方：

#### 偏差 1：ILP 的作用目标弄混了 (Chapter 4)

- **你的实现：** `ilp_solver.py` 是把所有 CNN 层分配给多个边缘节点（`n_nodes`），做层间（Inter-layer）负载均衡。
- **论文思路：** 论文的 ILP（Offloading Partitioner）仅仅是为了在**终端设备（端）\**和\**边缘服务器集群（边）\**之间找\**一个（或少量）切分点**。比如决定 Conv1、Conv2 留在本地手机上算，后面的全部打包卸载给边缘。边缘服务器集群在这一步被视为一个整体。
- **修正建议：** 你的 ILP 应该只做 `Device vs Edge Cluster` 的划分。

#### 偏差 2：DDPG 的应用维度弄混了 (Chapter 5)

- **你的实现：** 在 `adaptive_partitioner.py` 的 `apply_ratios_to_layers` 函数中，你拿 DDPG 输出的比例（如 `[0.4, 0.35, 0.25]`）去**分配层数**（比如 Node0 算前4层，Node1 算中间3层）。
- **论文思路：** DDPG（Parallel Partitioner）是用于 **层内（Intra-layer）切分**的！这个比例 `[0.4, 0.35, 0.25]` 应该用来**分配每一层的 Filter（通道）数量或神经元数量**，而不是分配层。
- **修正建议：** 删掉按层划分的逻辑，把 DDPG 输出的比例直接喂给你的 `tensor_divide_by_filter`。

### 🛠️ 下一步：如何将代码串联起来（Edge_LD 架构改造）

为了让你的 Edge_LD 项目真正跑通 AdapCP，你需要按照以下顺序重构调度逻辑（建议在类似 `scheduler.py` 或总控脚本中实现）：

#### 第一步：端侧 ILP 决策（决定卸载点）

在终端（或总控节点）运行简化的 ILP，决定将模型在哪一层切开。

Python

```python
# 伪代码逻辑
local_layers, edge_layers = ILP_Solver.solve(local_device_power, total_edge_power, bandwidth)
# 假设决策结果是：layer 1-5 在本地，layer 6-18 卸载
```

#### 第二步：边侧 DDPG 决策（决定并行比例）

针对卸载到边缘的层（如 layer 6-18），调用 `AdaptivePartitioner` 获取各个边缘节点的算力分配比例。

Python

```python
# 在 adaptive_partitioner.py 中调用
ratios = partitioner.get_partition_ratios(computing_power, network_bw)
# 例如返回 ratios = [0.5, 0.3, 0.2] (对应 Node A, Node B, Node C)
```

#### 第三步：数据与权重的 Filter Splitting（重点！）

利用 `ratios` 和你写的 `tensor_op.py`，**重写模型推理的前向传播（Forward）过程**。 在发送给数据节点（DataNode）之前，不是按 `Width` 切割了，而是按 `Channel` 切割。

Python

```python
# 结合 tensor_op.py 的逻辑改造
def dispatch_to_datanodes(input_tensor, ratios, n_nodes):
    N, C_out, H, W = input_tensor.shape
    
    # 按照 DDPG 给出的比例计算每个节点该分多少个 Channel
    channel_splits = [int(C_out * r) for r in ratios]
    # 修正由于取整导致的通道总数不匹配问题
    channel_splits[-1] = C_out - sum(channel_splits[:-1]) 
    
    splits = []
    start = 0
    for c in channel_splits:
        splits.append(input_tensor[:, start:start+c, :, :])
        start += c
        
    # 然后将 splits[0], splits[1], splits[2] 分别通过网络发送给对应的 DataNode
    return splits
```

#### 第四步：各节点独立计算与结果 Merge

边缘服务器（`server.py` 或 `datanode.py`）收到属于自己的那部分 Filter 和输入数据后，执行局部的卷积/全连接计算，并将结果传回。 调度器收到所有子结果后，使用你的 `merge_filter_tensor` 进行通道维度的拼接（`torch.cat(dim=1)`）。



需要核心纠正的地方（论文思路与你代码的偏差）
在你目前的 ilp_solver.py 和 adaptive_partitioner.py 中，存在两个与论文架构设计不匹配的地方：

偏差 1：ILP 的作用目标弄混了 (Chapter 4)
你的实现： ilp_solver.py 是把所有 CNN 层分配给多个边缘节点（n_nodes），做层间（Inter-layer）负载均衡。

论文思路： 论文的 ILP（Offloading Partitioner）仅仅是为了在终端设备（端）和边缘服务器集群（边）之间找一个（或少量）切分点。比如决定 Conv1、Conv2 留在本地手机上算，后面的全部打包卸载给边缘。边缘服务器集群在这一步被视为一个整体。

修正建议： 你的 ILP 应该只做 Device vs Edge Cluster 的划分。

偏差 2：DDPG 的应用维度弄混了 (Chapter 5)
你的实现： 在 adaptive_partitioner.py 的 apply_ratios_to_layers 函数中，你拿 DDPG 输出的比例（如 [0.4, 0.35, 0.25]）去分配层数（比如 Node0 算前4层，Node1 算中间3层）。

论文思路： DDPG（Parallel Partitioner）是用于 层内（Intra-layer）切分的！这个比例 [0.4, 0.35, 0.25] 应该用来分配每一层的 Filter（通道）数量或神经元数量，而不是分配层。

修正建议： 删掉按层划分的逻辑，把 DDPG 输出的比例直接喂给你的 tensor_divide_by_filter。

如何将代码串联起来（Edge_LD 架构改造）
为了让你的 Edge_LD 项目真正跑通 AdapCP，你需要按照以下顺序重构调度逻辑（建议在类似 scheduler.py 或总控脚本中实现）：

第一步：端侧 ILP 决策（决定卸载点）
在终端（或总控节点）运行简化的 ILP，决定将模型在哪一层切开。

伪代码逻辑

local_layers, edge_layers = ILP_Solver.solve(local_device_power, total_edge_power, bandwidth)

假设决策结果是：layer 1-5 在本地，layer 6-18 卸载

第二步：边侧 DDPG 决策（决定并行比例）
针对卸载到边缘的层（如 layer 6-18），调用 AdaptivePartitioner 获取各个边缘节点的算力分配比例。

在 adaptive_partitioner.py 中调用

ratios = partitioner.get_partition_ratios(computing_power, network_bw)

例如返回 ratios = [0.5, 0.3, 0.2] (对应 Node A, Node B, Node C)

第三步：数据与权重的 Filter Splitting（重点！）
利用 ratios 和你写的 tensor_op.py，重写模型推理的前向传播（Forward）过程。
在发送给数据节点（DataNode）之前，不是按 Width 切割了，而是按 Channel 切割。

结合 tensor_op.py 的逻辑改造

def dispatch_to_datanodes(input_tensor, ratios, n_nodes):
    N, C_out, H, W = input_tensor.shape

按照 DDPG 给出的比例计算每个节点该分多少个 Channel

channel_splits = [int(C_out * r) for r in ratios]

修正由于取整导致的通道总数不匹配问题

channel_splits[-1] = C_out - sum(channel_splits[:-1]) 

splits = []
start = 0
for c in channel_splits:
    splits.append(input_tensor[:, start:start+c, :, :])
    start += c

然后将 splits[0], splits[1], splits[2] 分别通过网络发送给对应的 DataNode

return splits    

第四步：各节点独立计算与结果 Merge
边缘服务器（server.py 或 datanode.py）收到属于自己的那部分 Filter 和输入数据后，执行局部的卷积/全连接计算，并将结果传回。
调度器收到所有子结果后，使用你的 merge_filter_tensor 进行通道维度的拼接（torch.cat(dim=1)）

---

### 解决方案：如何让固定维度的 VGG 跑动态切片的推理？

你不需要重写几千行的 `mydefine_VGG.py`。最优雅的做法是在边缘子节点上，绕过标准的 `nn.Conv2d`，直接使用 `torch.nn.functional.conv2d` 来**动态切片权重**。

你可以为子节点编写一个动态推理引擎，如下所示：

#### 方法：动态切片权重 (在子节点上执行)

当主节点给子节点分配任务时，不仅要发送完整的输入数据，还要告诉这个节点它该负责哪一部分的 Filter（例如：从第 0 个通道算到第 128 个通道）。

你可以新建一个类似 `SlicedVGGExecutor` 的类：

Python

```python
import torch
import torch.nn.functional as F

class SlicedVGGExecutor:
    def __init__(self, full_model):
        """
        传入你预训练好的完整 VGG 模型
        """
        self.model = full_model
        
    def execute_sliced_conv(self, input_tensor, layer_name, start_filter, end_filter):
        """
        动态提取部分卷积核进行计算
        """
        # 1. 从原模型中获取完整的权重和偏置
        # 假设你要跑 VGG 的第一层卷积 features[0]
        layer = dict(self.model.named_modules())[layer_name]
        
        full_weight = layer.weight
        full_bias = layer.bias
        stride = layer.stride
        padding = layer.padding
        
        # 2. 关键步骤：按 DDPG 算出的边界点，切分输出通道 (dim=0 是 out_channels 维度)
        sliced_weight = full_weight[start_filter:end_filter, :, :, :]
        
        if full_bias is not None:
            sliced_bias = full_bias[start_filter:end_filter]
        else:
            sliced_bias = None
            
        # 3. 使用 functional API 进行无状态的卷积计算
        # 注意：input_tensor 是完整的特征图
        output_tensor = F.conv2d(input_tensor, sliced_weight, sliced_bias, stride=stride, padding=padding)
        
        return output_tensor

    def execute_sliced_linear(self, input_tensor, layer_name, start_neuron, end_neuron):
        """
        对应论文的全连接层 Output Neuron Splitting (ONS)
        """
        layer = dict(self.model.named_modules())[layer_name]
        full_weight = layer.weight
        full_bias = layer.bias
        
        # 线性层的权重 shape 是 [out_features, in_features]
        sliced_weight = full_weight[start_neuron:end_neuron, :]
        sliced_bias = full_bias[start_neuron:end_neuron] if full_bias is not None else None
        
        output_tensor = F.linear(input_tensor, sliced_weight, sliced_bias)
        
        return output_tensor
```

### 🔄 改造你的系统流程 (重塑 Pipeline)

按照上述方法，你需要对你的主节点调度器 (`scheduler.py` / `tensor_op.py`) 和子节点 (`server.py`) 做如下改造：

**1. 主节点 (Master) 的分发逻辑不再切特征图：** 删掉 `tensor_op.py` 中导致特征图变薄的代码。主节点调用 DDPG 拿到切分边界（你代码里现成的 `get_channel_partition_points`）后，将**【完整的 Input Tensor + 你的起始边界索引】**打包发送给各个 DataNode。

Python

```python
# 主节点伪代码
ratios = partitioner.get_partition_ratios(computing_power, network_bw)
boundaries = partitioner.get_channel_partition_points(ratios, C_out=256) 
# boundaries 形如 [0, 128, 205, 256]

for i in range(datanode_num):
    start_filter = boundaries[i]
    end_filter = boundaries[i+1]
    # 把 full_input_tensor 和 start, end 一起通过 socket 发给对应的节点 i
    send_to_datanode(i, full_input_tensor, start_filter, end_filter)
```

**2. 子节点 (DataNode) 收到任务后：** 子节点收到完整的张量和索引后，调用上面的 `SlicedVGGExecutor`。由于每个子节点只截取了属于自己的那部分 `sliced_weight`，它算出来的张量通道数就会自动变小（例如算出来的输出只有 128 个通道）。 算完后，通过 Socket 传回主节点。

**3. 主节点 (Master) 合并：** 主节点收到 Node 0 传回的 `[N, 128, H, W]`，Node 1 传回的 `[N, 77, H, W]` 等，直接调用你的 `merge_filter_tensor`（也就是 `torch.cat`），就能完美拼凑出一个完整的 `[N, 256, H, W]` 张量，然后继续推给下一层！



### 解决方案：如何让固定维度的 VGG 跑动态切片的推理？

你不需要重写几千行的 `mydefine_VGG.py`。最优雅的做法是在边缘子节点上，绕过标准的 `nn.Conv2d`，直接使用 `torch.nn.functional.conv2d` 来**动态切片权重**。

你可以为子节点编写一个动态推理引擎，如下所示：

#### 方法：动态切片权重 (在子节点上执行)

当主节点给子节点分配任务时，不仅要发送完整的输入数据，还要告诉这个节点它该负责哪一部分的 Filter（例如：从第 0 个通道算到第 128 个通道）。

你可以新建一个类似 `SlicedVGGExecutor` 的类：

Python

```
import torch
import torch.nn.functional as F

class SlicedVGGExecutor:
    def __init__(self, full_model):
        """
        传入你预训练好的完整 VGG 模型
        """
        self.model = full_model
        
    def execute_sliced_conv(self, input_tensor, layer_name, start_filter, end_filter):
        """
        动态提取部分卷积核进行计算
        """
        # 1. 从原模型中获取完整的权重和偏置
        # 假设你要跑 VGG 的第一层卷积 features[0]
        layer = dict(self.model.named_modules())[layer_name]
        
        full_weight = layer.weight
        full_bias = layer.bias
        stride = layer.stride
        padding = layer.padding
        
        # 2. 关键步骤：按 DDPG 算出的边界点，切分输出通道 (dim=0 是 out_channels 维度)
        sliced_weight = full_weight[start_filter:end_filter, :, :, :]
        
        if full_bias is not None:
            sliced_bias = full_bias[start_filter:end_filter]
        else:
            sliced_bias = None
            
        # 3. 使用 functional API 进行无状态的卷积计算
        # 注意：input_tensor 是完整的特征图
        output_tensor = F.conv2d(input_tensor, sliced_weight, sliced_bias, stride=stride, padding=padding)
        
        return output_tensor

    def execute_sliced_linear(self, input_tensor, layer_name, start_neuron, end_neuron):
        """
        对应论文的全连接层 Output Neuron Splitting (ONS)
        """
        layer = dict(self.model.named_modules())[layer_name]
        full_weight = layer.weight
        full_bias = layer.bias
        
        # 线性层的权重 shape 是 [out_features, in_features]
        sliced_weight = full_weight[start_neuron:end_neuron, :]
        sliced_bias = full_bias[start_neuron:end_neuron] if full_bias is not None else None
        
        output_tensor = F.linear(input_tensor, sliced_weight, sliced_bias)
        
        return output_tensor
```

### 🔄 改造你的系统流程 (重塑 Pipeline)

按照上述方法，你需要对你的主节点调度器 (`scheduler.py` / `tensor_op.py`) 和子节点 (`server.py`) 做如下改造：

**1. 主节点 (Master) 的分发逻辑不再切特征图：** 删掉 `tensor_op.py` 中导致特征图变薄的代码。主节点调用 DDPG 拿到切分边界（你代码里现成的 `get_channel_partition_points`）后，将**【完整的 Input Tensor + 你的起始边界索引】**打包发送给各个 DataNode。

Python

```
# 主节点伪代码
ratios = partitioner.get_partition_ratios(computing_power, network_bw)
boundaries = partitioner.get_channel_partition_points(ratios, C_out=256) 
# boundaries 形如 [0, 128, 205, 256]

for i in range(datanode_num):
    start_filter = boundaries[i]
    end_filter = boundaries[i+1]
    # 把 full_input_tensor 和 start, end 一起通过 socket 发给对应的节点 i
    send_to_datanode(i, full_input_tensor, start_filter, end_filter)
```

**2. 子节点 (DataNode) 收到任务后：** 子节点收到完整的张量和索引后，调用上面的 `SlicedVGGExecutor`。由于每个子节点只截取了属于自己的那部分 `sliced_weight`，它算出来的张量通道数就会自动变小（例如算出来的输出只有 128 个通道）。 算完后，通过 Socket 传回主节点。

**3. 主节点 (Master) 合并：** 主节点收到 Node 0 传回的 `[N, 128, H, W]`，Node 1 传回的 `[N, 77, H, W]` 等，直接调用你的 `merge_filter_tensor`（也就是 `torch.cat`），就能完美拼凑出一个完整的 `[N, 256, H, W]` 张量，然后继续推给下一层！

---



1、我先不考虑精度损失的问题，ILP划分多少计算多少

2、加入AllReduce策略All-Gather





一、构建静态映射字典（Mapping Dictionary）

根据你在 `ilp_solver.py` 中设定的规则（池化层在 3, 6, 9, 12, 15），VGG13 的映射关系可以如下构建：

```
# 在 SlicedVGGExecutor 或配置文件中定义
VGG13_LAYER_MAPPING = {
    1: 'features_1_1',  # Conv2d
    2: 'features_1_2',  # Conv2d
    3: 'pool_1',        # MaxPool2d (不含权重，但需占位)
    4: 'features_2_1',  # Conv2d
    5: 'features_2_2',  # Conv2d
    6: 'pool_2',        # MaxPool2d
    7: 'features_3_1',  # Conv2d
    8: 'features_3_2',  # Conv2d
    9: 'pool_3',        # MaxPool2d
    10: 'features_4_1', # Conv2d
    11: 'features_4_2', # Conv2d
    12: 'pool_4',       # MaxPool2d
    13: 'features_5_1', # Conv2d
    14: 'features_5_2', # Conv2d
    15: 'pool_5',       # MaxPool2d
}

# 子节点动态获取权重的代码示例：
def get_layer_weights(model, layer_id):
    layer_name = VGG13_LAYER_MAPPING[layer_id]
    # 如果是池化层/激活层，直接返回 None，因为不需要切片权重
    if 'pool' in layer_name or 'relu' in layer_name:
        return None, None
    
    # 动态遍历获取对应模块
    layer_module = dict(model.named_modules())[layer_name]
    return layer_module.weight, layer_module.bias
```

二、加入论文中的AllReduce策略

1、 核心改造逻辑：引入同步屏障 (Synchronization Barrier)

为了实现 AllReduce，子节点不能再“算完就彻底结束”，而是要进入一个 **“计算 -> 提交切片 -> 阻塞等待 -> 接收全量张量 -> 继续计算”** 的循环。

你的 `PipelineScheduler` 需要增加一个“同步屏障”机制，把所有节点收集到的切片拼起来，再统一发回给它们。

2、 Scheduler (主节点) 的 AllReduce 改造

我们在你原有的 `scheduler.py` 中加入一个专门处理层间 AllReduce 的逻辑。

```
import threading
import time
import torch

class TaskInfo:
    def __init__(self, task_id, stage, datanode_id):
        self.task_id = task_id
        self.stage = stage # 当前计算的 layer_id
        self.datanode_id = datanode_id
        self.result_slice = None # 存储子节点传回的切片

class AllReduceScheduler:
    def __init__(self, datanode_num):
        self.datanode_num = datanode_num
        self.lock = threading.Lock()
        
        # 用于存放每一层各个节点返回的张量切片
        # 格式: { layer_id: { datanode_id: tensor_slice } }
        self.layer_slices = {} 
        
        # 用于通知主线程某一层已经收集完毕
        self.stage_events = {}

    def submit_slice(self, layer_id, datanode_id, tensor_slice):
        """
        步骤 1 (Gather): 子节点算完自己的部分通道后，提交给主节点
        """
        with self.lock:
            if layer_id not in self.layer_slices:
                self.layer_slices[layer_id] = {}
                self.stage_events[layer_id] = threading.Event()
            
            self.layer_slices[layer_id][datanode_id] = tensor_slice
            
            # 检查是否所有节点都提交了该层的切片
            if len(self.layer_slices[layer_id]) == self.datanode_num:
                self.stage_events[layer_id].set() # 唤醒 AllReduce 合并线程

    def wait_and_merge_all_reduce(self, layer_id):
        """
        步骤 2 & 3 (Merge & Broadcast): 主节点等待收集齐，合并，并准备广播
        """
        # 阻塞等待，直到当前层的所有节点都调用了 submit_slice
        if layer_id not in self.stage_events:
            with self.lock:
                self.stage_events[layer_id] = threading.Event()
                
        self.stage_events[layer_id].wait()

        # 开始合并 (使用你 tensor_op.py 里的逻辑)
        with self.lock:
            slices_dict = self.layer_slices[layer_id]
            # 必须按照 datanode_id 的顺序 (0, 1, 2...) 拼接，保证通道顺序正确
            ordered_slices = [slices_dict[i] for i in range(self.datanode_num)]
            
            # 在 Channel 维度 (dim=1) 进行拼接
            merged_tensor = torch.cat(ordered_slices, dim=1)
            return merged_tensor
```

3、 子节点 (DataNode) 的执行流改造

由于不考虑精度损失，ILP 给多少通道，节点就跑多少通道。你可以使用 `torch.nn.functional.conv2d` 强行截取权重并执行。

子节点的网络通信脚本（相当于你的 `client.py` 结合业务逻辑）需要变成一个 `while` 循环：

```
import torch
import torch.nn.functional as F
import socket

# 假设这个函数封装了你的 Socket 发送和接收逻辑
from network_op import send_tensor_to_master, receive_tensor_from_master

def run_datanode_allreduce_loop(datanode_id, my_weights, layers_to_run):
    """
    datanode_id: 当前节点ID (如 0, 1, 2)
    my_weights: 该节点预先加载好的对应每一层的权重切片 (start:end 已经切好)
    layers_to_run: ILP分配给边缘集群的层列表 (例如 [4, 5, 6])
    """
    
    # 接收初始的完整输入张量 (比如第 3 层的输出)
    current_input = receive_tensor_from_master() 
    
    for layer_id in layers_to_run:
        print(f"Node {datanode_id} 正在计算 Layer {layer_id}...")
        
        # 1. 本地计算 (不考虑精度，直接拿完整的输入算部分通道输出)
        weight_slice = my_weights[layer_id]['weight']
        bias_slice = my_weights[layer_id]['bias']
        stride = my_weights[layer_id]['stride']
        padding = my_weights[layer_id]['padding']
        
        # 计算出的 output_slice 通道数较少
        output_slice = F.conv2d(current_input, weight_slice, bias_slice, stride=stride, padding=padding)
        
        # 2. 发起 AllReduce 请求 (将切片发给主节点)
        send_tensor_to_master(datanode_id, layer_id, output_slice)
        
        # 3. 阻塞等待主节点的 AllReduce 广播结果
        print(f"Node {datanode_id} 等待 Layer {layer_id} 的 AllReduce 结果...")
        
        # current_input 被更新为拼接好的完整张量，进入下一轮循环
        current_input = receive_tensor_from_master() 
        print(f"Node {datanode_id} 收到 Layer {layer_id} 完整张量，准备进入下一层。")

    return current_input
```

4、 系统的整体运转视图 (主线程控制流)

在你的总控脚本中，主节点将扮演指挥官的角色，协调所有的 Socket 通信和 `AllReduceScheduler`：

```
# 总控节点伪代码
scheduler = AllReduceScheduler(datanode_num=3)
layers_to_run = [4, 5, 6] # ILP 决定卸载的层

# 把初始输入发给所有节点
broadcast_to_all_nodes(initial_tensor)

for layer_id in layers_to_run:
    # --- 这里是 Socket 异步接收线程做的事情 ---
    # 当 Socket 收到 Node i 发来的切片时，调用:
    # scheduler.submit_slice(layer_id, node_i, tensor_slice)
    # ----------------------------------------
    
    # 主线程阻塞，直到收集齐当前层 3 个节点的切片，并完成拼接
    full_merged_tensor = scheduler.wait_and_merge_all_reduce(layer_id)
    
    # 将拼接好的完整张量广播给所有节点，解除它们的阻塞，开始下一层！
    broadcast_to_all_nodes(full_merged_tensor)

print("边缘集群并行推理完成！")
```

三、我自己测试就行，不需要你来

四、其他新增改动

1、更改同步其他分支节点datanode

2、更改同步mydefineVGG16，但当前只考虑VGG13先不考虑VGG16的具体执行





🛠️namenode

~~~python
def compute_filter_boundaries(self, c_out):
    """根据DDPG比例计算Filter分割边界（3节点 → 4个边界）"""
    if self.current_ratios is None:
        raise ValueError("DDPG ratios not computed")

    boundaries = [0]
    # 修复1：遍历所有比例，不要切片砍掉！
    for r in self.current_ratios:
        next_bound = int(boundaries[-1] + c_out * r)
        boundaries.append(next_bound)
    
    # 修复2：最后强制等于总通道数，保证不越界（不覆盖中间值）
    boundaries[-1] = c_out
    
    self.current_boundaries = boundaries
    return boundaries
~~~

🛠️`network_op.py` 第 527-529 行

boundaries_bytes = b'%&%'.join([str(b).encode('utf-8') for b in filter_boundaries])







![image-20260418235950377](https://ajiebucket01.obs.cn-east-3.myhuaweicloud.com/typora_picture/fix-dir/2026/04/485af2879b4df26a20e2e92d3211d730.png)

Traceback (most recent call last):
  File "namenode_0_4_warm.py", line 390, in <module>
    run_adapcp_inference(namenode, adapcp_namenode, round_idx)
  File "namenode_0_4_warm.py", line 285, in run_adapcp_inference
    edge_output = adapcp_namenode.run_allreduce_inference(local_output)
  File "namenode_0_4_warm.py", line 191, in run_allreduce_inference
    _, transfer_time
  File "..\node_test\network_op.py", line 548, in collect_slice_from_datanode
    data_total_len = int(str(data_total_len, encoding='utf-8'))
ValueError: invalid literal for int() with base 10: ''

![image-20260419000003265](https://ajiebucket01.obs.cn-east-3.myhuaweicloud.com/typora_picture/fix-dir/2026/04/1440bd7f6d7ab95a3ad07488d7ef8ce0.png)

第 1 轮发生错误: Layer features_2_1 does not have weight (not a conv/linear layer)
Traceback (most recent call last):
  File "datanode_0_4_warm.py", line 164, in datanode_persistent
    node_start_filter, node_end_filter
  File "datanode_0_4_warm.py", line 73, in execute_sliced_layers
    current = self.executor.execute_sliced_conv(current, layer_id, s_f, e_f)
  File "..\VGG\tensor_op.py", line 985, in execute_sliced_conv
    raise ValueError(f"Layer {layer_name} does not have weight (not a conv/linear layer)")
ValueError: Layer features_2_1 does not have weight (not a conv/linear layer)

关闭 DataNode 0 的Socket连接
DataNode 0 AllReduce 模式已关闭



0同1

![image-20260419000035926](https://ajiebucket01.obs.cn-east-3.myhuaweicloud.com/typora_picture/fix-dir/2026/04/c8a694ed9e18d14706d5a27c996cf9a6.png)

[ReceiveInitial] From Master: layers 4-6, boundaries [0, 34, 35], input torch.Size([1, 64, 112, 112])
第 1 轮发生错误: list index out of range
Traceback (most recent call last):
  File "datanode_2_4_warm.py", line 143, in datanode_persistent
    node_end_filter = filter_boundaries[datanode_name + 1]
IndexError: list index out of range

关闭 DataNode 2 的Socket连接
DataNode 2 AllReduce 模式已关闭





---



这是修改后的代码，目前出现报错

主节点报错

--- Layer Group 5: Layers 16-18 ---

Traceback (most recent call last):

 File "namenode_0_4_warm.py", line 407, in <module>

  run_adapcp_inference(namenode, adapcp_namenode, round_idx)

 File "namenode_0_4_warm.py", line 302, in run_adapcp_inference

  edge_output = adapcp_namenode.run_allreduce_inference(local_output)

 File "namenode_0_4_warm.py", line 180, in run_allreduce_inference

  current_group_c_out = c_out_list[start_l - 1]

IndexError: list index out of range

子节点报错

[Node 0] 收到任务:

 Layers: 13-15

 Filter: [0:58]

 Input: torch.Size([1, 512, 14, 14])

 Layer 13: Conv2d [0:58], output torch.Size([1, 58, 14, 14])

 Layer 14: Conv2d [0:58], output torch.Size([1, 58, 14, 14])

 Layer 15: MaxPool2d, output torch.Size([1, 58, 7, 7])

[Node 0] 计算完成: torch.Size([1, 58, 7, 7]), 耗时: 0.011s

[SendSlice] To Master: Layer 15, Node 0, size torch.Size([1, 58, 7, 7])

[ReceiveMerged] From Master: next_layer=16, size torch.Size([1, 512, 7, 7])

[Node 0] 收到合并结果, next_layer=16

第 1 轮发生错误: invalid literal for int() with base 10: ''

Traceback (most recent call last):

 File "datanode_0_4_warm.py", line 135, in datanode_persistent

  broadcast_data = allreduce_datanode.receive_initial_broadcast()

 File "..\node_test\network_op.py", line 651, in receive_initial_broadcast

  data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

ValueError: invalid literal for int() with base 10: ''

我希望项目代码支持对全连接层的分布式计算



🛠️

`VGG13`

self.c_out_list = [64, 64, 64,  128, 128, 128,  256, 256, 256,   512, 512, 512,   512, 512, 512, 4096, 4096, num_classes]



代码在正常执行完成两轮后出现错误

主节点日志

============================================================

第 3 轮 AdapCP 推理 (AllReduce Filter Splitting)

============================================================



[ILP决策] Local: layers 1-3, Edge: layers 4-15

[时间估算] 本地: 0.092s, 传输: 0.032s, 边缘: 0.726s



===== 本地推理: Layers 1-3 =====

本地推理耗时: 0.187s, 输出尺寸: torch.Size([1, 64, 112, 112])



[DDPG决策] 分割比例: ['0.150', '0.783', '0.067']

[Filter边界] [0, 9, 59, 64]



===== AllReduce边缘推理: Layers 4-18 =====

层分组: [(4, 6), (7, 9), (10, 12), (13, 15), (16, 16), (17, 17), (18, 18)]



--- Layer Group 1: Layers 4-6 ---

当前组 (目标通道数 128) Filter 分割边界: [0, 19, 119, 128]

[Broadcast] Sent to Node 0: layers 4-6, boundaries [0, 19, 119, 128]

[Broadcast] Sent to Node 1: layers 4-6, boundaries [0, 19, 119, 128]

[Broadcast] Sent to Node 2: layers 4-6, boundaries [0, 19, 119, 128]

Traceback (most recent call last):

 File "namenode_0_4_warm.py", line 411, in <module>

  run_adapcp_inference(namenode, adapcp_namenode, round_idx)

 File "namenode_0_4_warm.py", line 303, in run_adapcp_inference

  edge_output = adapcp_namenode.run_allreduce_inference(local_output)

 File "namenode_0_4_warm.py", line 195, in run_allreduce_inference

  _, transfer_time

 File "..\node_test\network_op.py", line 555, in collect_slice_from_datanode

  data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

ValueError: invalid literal for int() with base 10: ''



子节点0

[Node 0] 收到任务:

 Layers: 18-18

 Filter: [0:0]

 Input: torch.Size([1, 4096])

[Node 0] 分到的通道/神经元数为0，跳过计算

[SendSlice] To Master: Layer 18, Node 0, size torch.Size([1, 0, 1, 1])

第 2 轮发生错误: buffer size must be a multiple of element size

Traceback (most recent call last):

 File "datanode_0_4_warm.py", line 160, in datanode_persistent

  merged_data = allreduce_datanode.receive_merged_tensor()

 File "..\node_test\network_op.py", line 753, in receive_merged_tensor

  recv_numpy = np.frombuffer(split_list[2], dtype=np.float32)

ValueError: buffer size must be a multiple of element size



关闭 DataNode 0 的Socket连接

DataNode 0 AllReduce 模式已关闭



子节点1

----- DataNode 1 第 3 轮推理开始 -----

[ReceiveInitial] From Master: layers 4-6, boundaries [0, 19, 119, 128], input torch.Size([1, 64, 112, 112])



[Node 1] 收到任务:

 Layers: 4-6

 Filter: [19:119]

 Input: torch.Size([1, 64, 112, 112])

 Layer 4: Conv2d [19:119], output torch.Size([1, 100, 112, 112])

 Layer 5: Conv2d [0:100], output torch.Size([1, 100, 112, 112])

 Layer 6: MaxPool2d, output torch.Size([1, 100, 56, 56])

[Node 1] 计算完成: torch.Size([1, 100, 56, 56]), 耗时: 0.198s

第 3 轮发生错误: [WinError 10053] 你的主机中的软件中止了一个已建立的连接。

Traceback (most recent call last):

 File "datanode_1_4_warm.py", line 185, in datanode_persistent

  allreduce_datanode.send_slice_to_master(end_layer, output_tensor)

 File "..\node_test\network_op.py", line 716, in send_slice_to_master

  self.datanode.datanode_socket.sendall(send_data)

ConnectionAbortedError: [WinError 10053] 你的主机中的软件中止了一个已建立的连接。



关闭 DataNode 1 的Socket连接

DataNode 1 AllReduce 模式已关闭

子节点2

正常执行完第三轮4-6推理等待后续







复现 **AdapCP** 的第四和第五章方法，实际上是构建一个“两阶段”的决策与执行系统 。作为一个正在进行相关课题研究的学长，我建议你按照“**底层机制 -> 宏观决策 -> 微观决策 -> 整体集成**”的逻辑来推进。

以下是为你整理的复现路线图：

## 第一阶段：基础分析与性能画像 (Profiling)

在编写算法之前，你需要获取模型在特定硬件上的基准数据。

- **硬件测速**：测量终端（如 Jetson Nano）和各边缘服务器的计算功率 $Q_e$ 和 $Q_s$ 。

- **网络画像**：测量节点间的平均带宽 $B^e$ 。

- **模型分析**：针对你的目标 CNN（如 VGG16 或 MobileNet），计算每一层的浮点运算量 $f_i$ 。

	- 卷积层（CL）计算公式：

		$$f_{i}=(2\times K_{i}^{h}\times K_{i}^{w}\times C_{in}-1)\times C_{out}\times H_{i}^{out}\times W_{i}^{out}$$

	- 全连接层（FL）计算公式：

		$$f_{i}=(2\times I-1)\times O$$


## 第二阶段：层内并行机制实现 (Mechanism Implementation)

复现第五章提到的物理拆分逻辑。这是算法决策后的“执行层”。

- **卷积层划分 (Filter Splitting)**：实现将输出通道维度切分的逻辑 。每个节点接收完整的输入特征图，但只计算一部分滤波器 

- **全连接层划分 (ONS)**：实现“输出神经元划分”逻辑 。每个节点保留完整的输入数据，只计算部分输出神经元的激活值 

- **合并操作 (Merge)**：编写一个高效的收集端逻辑，用于汇聚各服务器返回的子特征图或神经元片段 

## 第三阶段：层间划分决策 (Section 4: ILP Strategy)

实现“宏观”决策，确定哪些层放在本地，哪些卸载 。

- **建模**：基于变量 $x_{i,j}$（本地执行）和 $y_{i,j}$（卸载执行）构建 ILP 模型 

- **目标函数**：构建最小化总延迟的公式：

	$$T_{total}=Min(T_{comp}+T_{trans})$$

- **求解**：对于层数较少的 CNN，直接使用 ILP 求解器（如 PuLP 或 Gurobi）；对于深层网络，采用论文建议的 **凸松弛 (Convex Relaxation)** 方法，将变量设为 $[0, 1]$ 后再进行四舍五入 

## 第四阶段：层内比例决策 (Section 5: DRL Strategy)

实现“微观”决策，决定多台服务器各自承担的比例 

- **环境搭建**：构建强化学习环境，状态（State）包括服务器状态、带宽和 CNN 层信息 

- **DDPG 模型**：搭建 Actor 和 Critic 网络 

- **动作空间优化**：**关键点**是引入 **Dirichlet 分布** $\Phi_{i}\sim Dir(\phi)$，确保所有服务器分配比例之和为 $1$，并增强探索能力 

- **训练**：按照算法 2 (Algorithm 2) 进行迭代，利用奖励函数（基于用户预期时间和实际延迟）优化模型 

## 💡 复现建议

1. **先跑通物理拆分**：建议先手动指定一个划分点（比如 VGG16 的第 10 层），手动编写 Filter Splitting 逻辑，确保多台设备能协作推理出正确结果

2. **ILP 是基础**：第四章的 ILP 是整个框架的第一步（Offloading Partitioner），如果这一步选错切分点，后面的 DRL 优化空间会很小 

3. **注意冷启动**：论文提到单次冷启动决策可能需要 1-3 分钟 。在复现时，你可以先在离线环境下训练好 DRL 模型，再部署到实时的推理系统中

你目前在复现过程中，是卡在 ILP 的数学建模上，还是 DDPG 的网络编写上？



---



这篇论文提出的 **AdapCP** 框架旨在解决在资源受限的终端设备（如 Jetson Nano）上运行大型卷积神经网络（CNN）时的延迟和隐私问题 。其核心思想是将 CNN 推理任务巧妙地分给终端和多个边缘服务器共同完成 。

论文的 **第四节（Section 4）** 和 **第五节（Section 5）** 分别对应了该框架的两个关键阶段：**层间决策（Offloading Stage）** 和 **层内并行（Parallel Stage）** 。

## 第四节：层间划分的整数线性规划（ILP）

这一节解决的是“宏观”问题：**应该在 CNN 的哪一层切开，把哪一部分甩给边缘服务器处理？** 

### 1. 建模思路

为了找到最优切分点，作者建立了以下数学模型：

- **资源模型**：量化了边缘服务器的计算能力、终端设备的功率以及它们之间的网络带宽 $B^e$ 。

- **CNN 模型**：计算每一层所需的浮点运算量（FLOPs），以此推算在终端或服务器上的执行时间 。

	- 卷积层（CL）FLOPs 公式：

		$$f_{i}=(2\times K_{i}^{h}\times K_{i}^{w}\times C_{in}-1)\times C_{out}\times H_{i}^{out}\times W_{i}^{out}$$

	- 全连接层（FL）FLOPs 公式：

		$$f_{i}=(2\times I-1)\times O$$

### 2. 优化目标

作者将问题描述为一个 **整数线性规划（ILP）** 问题 。

- **核心变量**：使用二进制变量 $x_{i,j}$ 和 $y_{i,j}$ 表示某一段层是在本地执行还是卸载到服务器 

- **总目标**：最小化 **计算延迟** $T_{comp}$ 与 **传输延迟** $T_{trans}$ 之和 。

	$$T_{total}=Min(T_{comp}+T_{trans})$$

> **为什么要用 ILP？** 因为层间切分点的决策空间相对较小（CNN 的层数通常在几十到几百层），ILP 可以在有限时间内给出精确的最优解 。

## 第五节：层内并行的深度强化学习（DRL）

当第四节决定了哪些层要交给“边缘服务器群”后，第五节解决的是“微观”问题：**如何将这些层进一步拆分，让多个边缘服务器并行计算？** 

### 1. 不同层类型的并行策略

作者对比了多种拆分方法，找到了最有效率的方案：

- **全连接层（FLs）**：
	- **ONS（输出神经元划分）优于 INS（输入神经元划分）** 。
	- **原因**：ONS 允许在每个服务器上独立完成激活函数（如 ReLU），这样负数被截断为零，可以减少需要传输的数据量 。
- **卷积层（CLs）**：
	- **滤波器划分（Filter Splitting）优于 1D/2D 网格划分** 。
	- **原因**：网格划分（Grid Splitting）在处理像 MobileNet 这种深度可分离卷积时，会产生大量数据冗余和边界开销 。

### 2. 基于 DDPG 的自适应划分

由于每个服务器应该分担 **百分之多少** 的任务是一个连续的决策问题，传统的算法很难处理这种高维空间 。因此，作者采用了 **深度确定性策略梯度（DDPG）** 算法 。

- **Dirichlet 分布**：作者在 DDPG 中引入了狄利克雷分布，这能确保分配给各服务器的任务比例之和始终为 $1$，同时提高探索效率 。
- **动作（Action）**：向量 $\Phi_{i}$ 代表每一层在各个服务器上的切分比例 。
- **奖励（Reward）**：如果能在用户预期时间内完成推理，则给予奖励；否则惩罚。同时鼓励系统保持分区稳定 。

## 总结：两者的配合

| **阶段**                | **解决问题**                                     | **采用算法**   | **粒度**                      |
| ----------------------- | ------------------------------------------------ | -------------- | ----------------------------- |
| **第四节 (Offloading)** | 决定 CNN 哪一段放在终端，哪一段甩给边缘。        | **ILP**        | 粗粒度（层级）                |
| **第五节 (Parallel)**   | 决定卸载到边缘的任务如何在多台服务器间分配比例。 | **DDPG (DRL)** | 细粒度（层内神经元/滤波器级） |

这种“先定大方向，再磨细分工”的两步走策略，使推理速度比传统方案平均提升了 **2.21 倍** 。

你对这两个算法中的具体数学式，或者是它们在 Jetson Nano 上的实验表现更感兴趣吗？



---

cd d:\Graduate\WorkBench\work_1\Edge_LD\node_test



> 1、数据集CIFAR-10 （精度没有损失，VGG跑的就行
>
> https://www.kaggle.com/datasets/pankrzysiu/cifar10-python/code?datasetId=11102&searchQuery=VGG
>
> 2、不要提端边协同（找个老师较忙的时候问可不可行
>
> 3、如果到时候老师问为什么能提升这么快怎么办（是不是还得优化一下他的算法



1、VGG模型没有问题

这个代码实现的VGG13模型有没有问题，是不是少了部分层的relu操作

🛠️

- conv后少了relu
- 三层全连接层需要改动

self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
self.classifier_1 = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(),
)
self.classifier_2 = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(),
)
self.classifier_3 = nn.Sequential(
    nn.Linear(4096, num_classes),
)

|      |                                                              |
| ---- | ------------------------------------------------------------ |
| 16   | nn.AdaptiveAvgPool2d((7, 7))->torch.flatten(x, 1)->nn.Linear(512 * 7 * 7, 4096)->nn.ReLU(True)->nn.Dropout() |
| 17   | nn.Linear(4096, 4096)->nn.ReLU(True)->nn.Dropout()           |
| 18   | nn.Linear(4096, num_classes)                                 |

- [x] 卷积 relu
- [x] 全连接 average+relu+fc

2、确认层分割算法和层内分割算法

- [x] 层分割优化 all-reduce

ILP避免端边通信（手机端服务器边），边边通信合并每次卷积（论文环境不一样

all-reduce指的就是边边通讯，保证每一次卷积的输入是整个张量（无损

**不存在all-reduce算法重新优化层分割**

**卷积输入是全部维度，输出才分割**



❓是否误解源码了，存在部分文件agent无法读取（应该没有，并不完整




- [x] 层分割算法 ILP

实际上无法实现必要边与边通信，那就在本基础上实现必要通信

**p2p实现边边通信**



- [x] 层内分割算法 DRL DDPG

> ❓(1, 512, 7, 7)执行flatten->fc4096->relu->dropout->fc4096->relu->dropout->fc100 和(1, 128, 112, 112)切分为[1, 174, 7, 7] [1, 170, 7, 7] [1, 168, 7, 7], 分别进行[1, 174, 7, 7] flatten->fc1395->relu->dropout->fc1395->relu->dropout->fc34; [1, 170, 7, 7] flatten->fc1360->relu->dropout->fc1360->relu->dropout->fc33; [1, 168, 7, 7]flatten->fc1341->relu->dropout->fc1341->relu->dropout->fc33 再进行合并结果一样么
>
> （要么就直接返回主节点算， 没意义，还可能可能传输延时更高，分布算通信延时没有很高）
>
> 
>
> 简单直接的结论是：**你提出的切分方案将一个“全连接（Dense）”结构变成了“分块对角（Block-Diagonal）”结构。** 神经元之间的信息交叉（Cross-talk）被强行切断了
>
> 
>
> 考虑到你正在研究 **VGG 层的分布式切分** 和 **边缘节点协作**，这种“按通道硬切分”FC 层通常有两种处理思路：
>
> 思路 A：如果你追求“数学等效”
>
> 如果你希望分布式执行后的结果和单机运行 **完全一样**，你不能直接把 FC 层切成独立的小 FC。
>
> - **正确做法：** 采用 **模型并行（Model Parallelism）** 中的纵向切分。每个节点计算 $X_i \cdot W_i$，然后将结果（Partial Sum）发送到主节点进行累加：$Y = \sum (X_i \cdot W_i) + b$。
> - **缺点：** 通信开销大，因为每个节点需要向主节点发送 4096 维的向量。
>
> 思路 B：如果你追求“通信优化”（即你提出的做法）
>
> 你提出的做法实际上是一种 **模型压缩/剪枝** 策略。
>
> - **适用场景：** 如果你愿意牺牲一点精度（Accuracy），来换取极低的通信成本（每个子节点只发送约 33 维的结果）。
> - **必要步骤：** 如果按你这种方式切分，你**不能**直接加载原生的 VGG 权重，因为权重维度对不上。你必须**重新训练**这个切分后的网络，让模型习惯这种“分组”的信息流。

3、测时间（可能没有必要，论文就是这样实现的

- [ ] 你可以判断一下主节点从开始执行到结束，分别有哪些步骤有时间开销，统计一下



4、节点和VGG16

- [ ] node=1、node=2
- [ ] VGG16
- [ ] width=448

5、拿原始EdgeLD跑CIFAR-10

- [ ] AdapCP准确率 79.8%；原始EdgeLD准确率 __%

[VGG16 Pre-trained Architecture (beginner)](https://www.kaggle.com/code/vortexkol/vgg16-pre-trained-architecture-beginner/notebook)



老师您好，您之前考虑到我时间紧张，建议我选择复现两个经典方法。这几天我还是尝试复现了那篇 25 年的论文方法，但复现得出的时间结果，并没有比 EdgeLD 提出的 ODBP 方法更优。所以想向您请教，这种情况下，这篇 25 年论文的方法是否还能作为对比方法？还是说按照您之前的建议，仅复现两个经典方法即可？

另外跟您说明一下，那篇 25 年论文提供的代码仓库并不完整，目前我使用的代码是自己复现的，之后我也咨询过 AI，确认我复现的代码符合论文中的具体实现要求。



![image-20260421140154331](https://ajiebucket01.obs.cn-east-3.myhuaweicloud.com/typora_picture/fix-dir/2026/04/8aaac2f1b2b16ce30e157a7184a517fd.png)



AdapCP这种方法避免不了**层间通信开销大**(通信次数甚至比ocbp还频繁)并且**卷积时间没有明显降低**, OCBP和PABC避免了频繁的通信同时也兼顾了大张量的卷积开销, 因而时间理应更快

AdapCP 

1 他是通过**划分通道**以削减通信时间和卷积时间, **考虑多样的卷积模型**

但是这种划分**在每一层卷积层计算**时为了保证精度**不可避免的需要通信**(论文通过边边通信)使得卷积的输入张量为合并的整张张量, 这也使得他的通信次数更多

并且由于**卷积层计算输入是整张张量**卷积时间也并没有明显的降低

OCBP和PABC

1 是**针对VGG模型**进行的优化 

2 使用1grid划分 **卷积输入和输出都是被划分的小张量**, 卷积时间更低, 通过适当扩展避免精度损失

3 使用1grid划分 **通信次数更低**, 只需要在每个VGG Block块才需要通讯



==传输时间 卷积时间??==

AdapCP 1主1从的话, 那就是单纯吧任务从VGG第四层开始丢给子节点执行到最后(没有通信, 只有完整的卷积计算), 跑没有意义



他的设备性能比我的更强



基于终端与边缘服务器的带宽、算力配置以及CNN各层特征信息（如运算量与数据量），通过构建总延迟最小化目标并进行凸优化求解的整数线性规划（ILP）层间卸载算法，主要优化了端边设备间的粗粒度（层级别）切分点与卸载策略，最大限度地降低了整体推理与数据传输的总时间开销。



稳定环境配置基本相同, 所以基本都是按比例均分

基于边缘计算网络的动态状态（如各节点状态、节点间带宽及当前处理层的维度），通过结合 Dirichlet 分布来高效探索连续动作空间的 Actor-Critic 深度强化学习（DRL）的 DDPG 层内并行拆分算法，主要优化了高维复杂参数空间下的细粒度切分比例（即滤波器或神经元的分配），在确保满足用户预期时间的同时最大化了多台边缘服务器并行计算的效率。

==如果子节点在执行ILP分配的连续卷积层时，需要逐层进行DDPG确定通道分割比例么==



### ILP和DRL（DDPG）需要的输入

在项目代码中，ILP（层间卸载决策）和 DRL/DDPG（层内分割比例决策）的输入主要分为**模型结构信息（静态）**和**环境资源状态（动态）**两部分。

以下是具体的输入参数解析，主要基于 `ilp_solver.py`、`adaptive_partitioner.py` 以及 `namenode_0_4_warm_.py` 中的调用逻辑：

1. ILP (整数线性规划 / 穷举搜索) 的输入

ILP 的目标是找到一个最优的层间切分点（Split Layer），决定哪些层在端侧（Device）运行，哪些层在边缘集群（Edge Cluster）运行。

A. 静态网络结构信息 (通过 `add_layer` 提前输入)

在运行求解器之前，必须先将 CNN 每一层的特征“喂”给 `OffloadingPartitioner`。需要的输入包括：

- **`layer_id`**: 层的序号（如 1, 2, 3...）。
- **`flops`**: 该层的浮点运算次数（用于估算计算延迟）。
- **`output_size`**: 该层输出特征图的数据量大小（用于估算传输延迟）。
- **其他拓扑参数**: 是否为池化层 `is_maxpool`、输入/输出通道数 `c_in`/`c_out`、特征图高宽 `h`/`w` 等。

B. 动态资源状态 (通过 `solve_offloading_point` 输入)

在实际推理或决策时，需要传入当前的环境资源状态：

- **`device_power`**: 端侧设备的计算能力参数，格式为元组 `(a, b)`。代码中使用线性模型预估计算时间：$Time = a \times \text{FLOPs} + b$。例如代码中的 `(6.24e-11, 1.97e-2)`。
- **`edge_power`**: 边缘集群的整体计算能力参数，格式同上 `(a, b)`。
- **`bandwidth`**: 端侧设备与边缘集群之间的网络带宽（单位：bps），例如 `100e6` (100 Mbps)。
- **`min_local_layers` / `max_local_layers`**: 约束条件，限制最少或最多在本地执行的层数。

2. DRL / DDPG 的输入

DRL（在代码中表现为 `AdaptivePartitioner` 和 `DDPGAgentDirichlet`）的目标是决定把边缘侧计算的层切分成多少份，以及分配给各个边缘节点的比例。

A. 强化学习的“状态 (State)”输入

在每次决策前，调用 `get_partition_ratios` 需要传入环境的实时状态，这些输入会在内部被 `_build_state` 拼接并归一化成一个一维状态向量喂给神经网络：

- **`computing_power`**: 边缘节点的计算能力列表。包含了所有参与计算的边缘节点参数，格式为 `[(a1, b1), (a2, b2), ...]`。
- **`network_bw`**: 边缘节点的网络带宽列表，表示各个节点与主控节点间的带宽，格式为 `[bw1, bw2, ...]`。
- **`current_load`**: （可选参数）各边缘节点当前的负载情况列表，格式为 `[load1, load2, ...]`。代码中如果没有传入，默认以 `0.0` 填充。

> **注**：对于 $N$ 个节点，状态向量的维度默认是 $4N$（每个节点贡献 $a, b, \text{bw}, \text{load}$ 四个归一化特征）。代码中设定 `state_dim = 12` 正好对应 3 个节点。

B. 强化学习的“超参数”输入

- **`epsilon`**: 探索因子。在 `namenode_0_4_warm_.py` 中，预热阶段（Warm-up）设为 `0.5`，使 Agent 能利用 Dirichlet 分布和 OU 噪声去探索未知空间；正式验证阶段设为 `0.0`，此时 DDPG 直接输出确定性的最优比例。

C. 环境反馈的“奖励 (Reward)”输入

虽然这不是决策时的输入，但它是 DDPG 网络**更新参数（学习）的必备输入**：

- **`latency`**: 执行完一轮推理后的真实总耗时。通过 `record_inference_result(latency)` 传入，代码内部会将其转换为 Reward：$Reward = C - \log(\max(\text{latency}, 1e-5))$。

总结对照表

| **算法** | **目标**                              | **必须的运行时输入参数**                                 |
| -------- | ------------------------------------- | -------------------------------------------------------- |
| **ILP**  | 决定端侧与边缘的**切分点** (层间)     | `device_power`, `edge_power`, `bandwidth`                |
| **DDPG** | 决定各边缘节点的**工作量比例** (层内) | `computing_power` (列表), `network_bw` (列表), `epsilon` |

了解了这些输入后，你是否打算将这段代码迁移到你自己的模型（比如 ResNet 或 MobileNet）上？如果是的话，我可以进一步讲解如何快速提取新模型的 FLOPs 和特征图尺寸来构造这些算法的初始输入。
