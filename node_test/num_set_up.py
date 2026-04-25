import torch

# ========== 全局配置（在此处修改）==========
MODEL_TYPE = 'VGG16'      # 'VGG13' 或 'VGG16'
data_frame = 1
INPUT_WIDTH = 224         # 96, 224, 448
datanode_num_temp = 3     # 1, 2, 3

sample_tenosr = torch.randn(data_frame, 3, INPUT_WIDTH, INPUT_WIDTH)

# 根据 MODEL_TYPE 导入对应的 VGG 模型类
if MODEL_TYPE == 'VGG16':
    from VGG.mydefine_VGG16 import VGG_model as VGG_model_class
else:
    from VGG.mydefine_VGG13 import VGG_model as VGG_model_class

# 初始化模型并导出关键属性
inference_model = VGG_model_class()
c_out_list = inference_model.get_c_out()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
maxpool_layer = inference_model.get_maxpool_layer()

# 计算卷积层层号 (非池化层的层)
pool_layers_set = set(maxpool_layer)
conv_layers = [i for i in range(1, conv_length + 1) if i not in pool_layers_set]


class Num_set_up(object):
    def __init__(self):
        self.namenode_num = 1
        self.datanode_num = datanode_num_temp

    def get_model_type(self):
        return MODEL_TYPE

    def get_input_width(self):
        return INPUT_WIDTH

    def get_c_out_list(self):
        return c_out_list

    def get_conv_length(self):
        return conv_length

    def get_total_length(self):
        return total_length

    def get_maxpool_layer(self):
        return maxpool_layer

    def get_conv_layers(self):
        return conv_layers

    def get_inference_model(self):
        return inference_model

    def get_namenode_num(self):
        return 1

    def get_datanode_num(self):
        return datanode_num_temp
