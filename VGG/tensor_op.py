import sys
sys.path.append("../..")
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
from VGG.mydefine_VGG16 import VGG_model

# 针对设备计算能力 和 网络通信状态的
def tensor_divide_by_computing_network_and_fill(original_tensor, datanode_num = 1, cross_layer = 1, in_chanel = 1,
                                                out_chanel = 1, kernerl_size = 3, computing_power = [], network_state = []):
    a = 0



# 定义一个tensor直接分割, original_tensor默认为4维, 划分后的[start, end],含start，不包含end
def tensor_divide(original_tensor, divide_num = 1):
    divided_tensor = []
    divide_record = np.zeros((divide_num, 2), dtype=np.int)
    if divide_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # [0]start,[1]end
        for it in range(divide_num):
            start = int(width / divide_num * it)
            end =  int( width / divide_num * (it + 1) )
            print("[ %d, %d]" % (start, end))
            # 记录
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 根据start得到tensor
            temp_tensor = original_tensor[:, :, :, start:end]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record

# 定义一个tensor直接分割, original_tensor默认为4维, 划分后的[start, end],含start，不包含end
def tensor_divide_and_fill(original_tensor, datanode_num = 1, cross_layer = 1):
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # [0]start,[1]end
        for it in range(datanode_num):
            start = int(width / datanode_num * it)
            end =  int( width / datanode_num * (it + 1) )
            print ("[ %d, %d]" %(start, end))
            # 记录
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start:int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record


# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_fill(original_tensor, datanode_num = 1, cross_layer = 1, computing_power=[]):
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num+1):
            sum_computing_power.append(sum(computing_power[0:i]))
        # [0]start,[1]end
        for it in range(datanode_num):
            start = int(sum_computing_power[it]/total_computing_power * width)
            end = int(sum_computing_power[it+1]/total_computing_power * width)
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start:int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record


# #############################################################################################################
# 时间估计
def get_prediction_time(datanode_num = 0, index = 0, length = 0, cross_layer = 1, computing_a = [],
                        computing_b = [], network_state = [], input_param = [], c_out = 0):
    input_number, c_in, height, width = input_param
    if c_out == 0:
        c_out = c_in
    else:
        c_out = c_out
    kernel = 3
    # 计算 FLOPs
    FLOPs = input_number * 2 * height * length * c_out * (kernel * kernel * c_in + 1)
    # 计算时间
    comp_time = computing_a[index] * FLOPs + computing_b[index]
    # 通信开销
    comm_data = input_number * c_in * height * 4.0 / network_state[index]
    comm_time = 0
    # 判断是否是边界
    if index == 0 or index == datanode_num - 1:
        comm_time = comm_data * (cross_layer + 2 * length)
    # 中间情况
    else:
        comm_time = 2 * comm_data * (cross_layer + length)
    prediction_time = comp_time + comm_time
    return prediction_time



# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network(original_tensor, datanode_num = 1, cross_layer = 1,
                                        computing_power = [], computing_a = [], computing_b = [], network_state = [], c_out = 0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0 : i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it+1]/total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num = datanode_num, index = it, length = length[it], cross_layer = cross_layer, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while(True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if ( diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num = datanode_num, index = index_max, length = length[index_max], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num = datanode_num, index = index_min, length = length[index_min], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)
        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start : int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record

# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network_pooled(original_tensor, datanode_num=1, cross_layer=1,
                                           computing_power=[], computing_a=[], computing_b=[], network_state=[],
                                           c_out=0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0: i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it + 1] / total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num=datanode_num, index=it, length=length[it],
                                                      cross_layer=cross_layer, computing_a=computing_a,
                                                      computing_b=computing_b, network_state=network_state,
                                                      input_param=input_param, c_out=c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while (True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if (diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num=datanode_num, index=index_max,
                                                             length=length[index_max], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num=datanode_num, index=index_min,
                                                             length=length[index_min], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)

        # ===================== 新增优化：保证每个切分width为偶数 =====================
        # 调整length数组，确保每个元素都是偶数，且总和不变
        total_length = sum(length)
        # 遍历调整每个长度为偶数
        for i in range(datanode_num):
            if length[i] % 2 != 0:
                length[i] -= 1  # 奇数减1变为偶数，保证最小幅度调整
        # 补偿因减1损失的长度，保证总宽度不变（偶数补偿，不破坏偶数性）
        compensate = total_length - sum(length)
        idx = 0
        while compensate > 0:
            length[idx] += 2
            compensate -= 2
            idx = (idx + 1) % datanode_num
        # ==========================================================================

        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start: int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer): end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer): int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record

# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network_pabc(original_tensor, datanode_num=1, cross_layer=1,
                                           computing_power=[], computing_a=[], computing_b=[], network_state=[],
                                           c_out=0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0: i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it + 1] / total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num=datanode_num, index=it, length=length[it],
                                                      cross_layer=cross_layer, computing_a=computing_a,
                                                      computing_b=computing_b, network_state=network_state,
                                                      input_param=input_param, c_out=c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while (True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if (diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num=datanode_num, index=index_max,
                                                             length=length[index_max], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num=datanode_num, index=index_min,
                                                             length=length[index_min], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)

        # ===================== 优化：保证每个切分 width 为 4 的倍数 =====================
        total_length = sum(length)
        # 第一步：把每个长度向下取为 4 的倍数
        for i in range(datanode_num):
            length[i] = (length[i] // 4) * 4  # 向下取 4 的倍数

        # 第二步：补偿总长度，保证总和不变，且每个块仍然是 4 的倍数
        compensate = total_length - sum(length)
        idx = 0
        while compensate > 0:
            length[idx] += 4  # 每次 +4，保持是 4 的倍数
            compensate -= 4
            idx = (idx + 1) % datanode_num
        # ==========================================================================

        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        divide_layer = cross_layer + 2
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start: int(end + divide_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - divide_layer): end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - divide_layer): int(end + divide_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor(divided_tensor = [], divide_record = [], cross_layer = 1):
    '''
    :param divided_tensor: 需要合并的tensor
    :param divide_record:  之前拆分位置的记录
    :return: 合并后的tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0
    if length == 1:
        return divided_tensor[0][:, :, :, :]
    merged_tensor = 0
    for it in range(length):
        if it == 0:
            # 最左侧
            merged_tensor = divided_tensor[it][:, :, :, 0:-cross_layer]
        elif it == length -1:
            # 最右侧
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, cross_layer:]), 3)
        else:
            # 中间非边界
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, cross_layer: -cross_layer]), 3)
    return merged_tensor

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor_pooled(divided_tensor = [], divide_record = [], cross_layer = 1):
    '''
    :param divided_tensor: 需要合并的tensor
    :param divide_record:  之前拆分位置的记录
    :return: 合并后的tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0
    if length == 1:
        return divided_tensor[0][:, :, :, :]
    merged_tensor = 0
    # ===================== 新增优化：切分合并保证大小 ============================
    divide_layer = cross_layer - 1
    # ===========================================================================
    for it in range(length):
        if it == 0:
            # 最左侧
            merged_tensor = divided_tensor[it][:, :, :, 0:-divide_layer]
        elif it == length -1:
            # 最右侧
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, divide_layer:]), 3)
        else:
            # 中间非边界
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, divide_layer: -divide_layer]), 3)
    return merged_tensor

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor_pabc(divided_tensor=[], divide_record=[], cross_layer=1):
    '''
    单纯拼接 divided_tensor 中的所有张量，在第4维（dim=3）合并
    :param divided_tensor: 需要合并的 tensor 列表
    :param divide_record:  无用参数（保留兼容）
    :return: 拼接后的完整 tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0

    # 直接在 dim=3 拼接所有张量，不做任何裁剪/切片
    merged_tensor = torch.cat(divided_tensor, dim=3)
    return merged_tensor

# 聚合 差分传输 的tensor，暂时不需要写
def merge_part_tensor(divided_tensor = [], divide_record = [], cross_layer = 1):
    return 0

# 用于datanode的差值交换，拆开计算结果tensor，分为  saved_tensor, divied_tensor
def divied_middle_output(input_tensor = 0, datanode_num = 1, datanode_name = 0, cross_layer = 1):
    saved_tensor = torch.rand(1, 1, 1, 1)
    divied_tensor = []
    # 最左
    if datanode_name == 0 :
        saved_tensor = input_tensor[:, :, :, 0 : -cross_layer]
        divied_tensor.append(  input_tensor[:, :, :, -cross_layer : ]  )
    # 最右
    elif datanode_name == datanode_num - 1:
        saved_tensor = input_tensor[:, :, :, cross_layer : ]
        divied_tensor.append(  input_tensor[:, :, :, 0 : cross_layer ]   )
    # 中间
    else:
        saved_tensor = input_tensor[:, :, :, cross_layer : -cross_layer]
        divied_tensor.append(  input_tensor[:, :, :, 0 : cross_layer]  )
        divied_tensor.append(  input_tensor[:, :, :, -cross_layer : ])
    return saved_tensor, divied_tensor


# 输入一个tensor，得到这个tensor的比特长度，传输数据时统计
def get_tensor_bytes_length( input_tensor ):
    input_num, input_chanel, height, width = input_tensor.size()
    numbers = input_num * input_chanel * height * width
    bytes_length = int(numbers * 4)
    return bytes_length

# 统计卷积运算的 FLOPS
def get_conv_tensor_flops(in_chanel = 1, out_chanel = 1, kernel_size = 3, input_height = 1, input_width = 1):
    return 2 * input_height * input_width * out_chanel * (in_chanel * kernel_size * kernel_size + 1)
# 统计全连接层的 FLOPS
def get_fully_tensor_flops(input = 1, output = 1):
    return output * (2 * input - 1)

# 主函数
# if __name__ == "__main__":
#     width = 224
#     num = 3
#     VGG16 = VGG_model()
#     input = torch.rand(1, 3, width, width)
#     computing_power = [4, 1, 4, 8, 7, 4, 4]
#     temp = [1, 2, 4, 5, 6]
#
#     divided_tensor, divide_record = tensor_divide_by_computing_and_fill(input, num, cross_layer = 2, computing_power = computing_power)
#     # 测试卷积后是否相同
#     output_1 = VGG16(input, 1, 2)
#     print (output_1.size())
#     output_tensor = []
#     for i in range(num):
#         output_tensor.append(VGG16(divided_tensor[i], 1, 2))
#     print ( len(output_tensor) )
#     merged_tensor = merge_total_tensor(output_tensor, divide_record, cross_layer = 2)
#     print(torch.equal(output_1, merged_tensor))


# ==================== AdapCP: Filter/Neuron Splitting ====================

def tensor_divide_by_filter(input_tensor, n_nodes, overlap=0):
    """
    按filter维度均匀分割张量 (AdapCP层内分割)
    input_tensor: [N, C_out, H, W]
    n_nodes: 分割节点数
    overlap: 重叠通道数（用于数据汇合）

    返回: List of tensor splits
    """
    N, C_out, H, W = input_tensor.shape

    if C_out < n_nodes:
        return [input_tensor]

    channels_per_node = C_out // n_nodes
    remainder = C_out % n_nodes
    splits = []

    start = 0
    for i in range(n_nodes):
        extra = 1 if i < remainder else 0
        end = start + channels_per_node + extra

        if overlap > 0 and i > 0:
            actual_start = max(0, start - overlap)
        else:
            actual_start = start

        if overlap > 0 and i < n_nodes - 1:
            actual_end = min(C_out, end + overlap)
        else:
            actual_end = end

        splits.append(input_tensor[:, actual_start:actual_end, :, :])
        start = end

    return splits


def tensor_divide_by_filter_ratios(input_tensor, channel_splits, overlap=0):
    """
    按指定通道分割点分割张量 (AdapCP Filter Splitting - 按比例分割)

    参数:
        input_tensor: [N, C_out, H, W] 输入张量
        channel_splits: [c1, c2, c3, ...] 各节点处理的通道数列表
        overlap: 重叠通道数（用于数据汇合）

    返回:
        splits: List of tensor splits

    示例:
        input_tensor: [N, 256, H, W]
        channel_splits: [128, 77, 51]
        返回3个tensor: [N, 128, H, W], [N, 77, H, W], [N, 51, H, W]
    """
    N, C_out, H, W = input_tensor.shape

    if len(channel_splits) == 0:
        return [input_tensor]

    total_assigned = sum(channel_splits)
    if total_assigned != C_out:
        raise ValueError(f"Channel splits sum {total_assigned} != total channels {C_out}")

    splits = []
    start = 0
    for i, c in enumerate(channel_splits):
        end = start + c

        if overlap > 0 and i > 0:
            actual_start = max(0, start - overlap)
        else:
            actual_start = start

        if overlap > 0 and i < len(channel_splits) - 1:
            actual_end = min(C_out, end + overlap)
        else:
            actual_end = end

        splits.append(input_tensor[:, actual_start:actual_end, :, :])
        start = end

    return splits


def tensor_divide_by_filter_boundaries(input_tensor, boundary_points, overlap=0):
    """
    按通道边界点分割张量 (AdapCP Filter Splitting - 按边界点分割)

    参数:
        input_tensor: [N, C_out, H, W] 输入张量
        boundary_points: [0, 64, 128, 192, 256] 通道边界点
        overlap: 重叠通道数（用于数据汇合）

    返回:
        splits: List of tensor splits
    """
    if len(boundary_points) < 2:
        return [input_tensor]

    C_out = input_tensor.shape[1]
    channel_splits = []
    for i in range(len(boundary_points) - 1):
        channel_splits.append(boundary_points[i + 1] - boundary_points[i])

    return tensor_divide_by_filter_ratios(input_tensor, channel_splits, overlap)


def merge_filter_tensor(splits, overlap=0):
    """
    合并filter分割的结果 (AdapCP层内分割汇合)

    参数:
        splits: List of tensor splits，每个tensor为[N, C_i, H, W]
        overlap: 重叠通道数

    返回:
        merged_tensor: [N, sum(C_i), H, W]
    """
    if len(splits) == 1:
        return splits[0]

    if overlap == 0:
        return torch.cat(splits, dim=1)

    n_nodes = len(splits)
    merged_channels = []

    for i, split in enumerate(splits):
        if i == 0:
            merged_channels.append(split)
        elif i == n_nodes - 1:
            merged_channels.append(split[:, overlap:, :, :])
        else:
            merged_channels.append(split[:, overlap:-overlap, :, :])

    return torch.cat(merged_channels, dim=1)


def tensor_divide_by_neurons(weight_tensor, n_nodes):
    """
    按神经元维度分割全连接层权重 (AdapCP层内分割)
    weight_tensor: [out_features, in_features]
    n_nodes: 分割节点数

    返回: List of weight splits
    """
    out_features = weight_tensor.shape[0]

    if out_features < n_nodes:
        return [weight_tensor]

    neurons_per_node = out_features // n_nodes
    remainder = out_features % n_nodes
    splits = []

    start = 0
    for i in range(n_nodes):
        extra = 1 if i < remainder else 0
        end = start + neurons_per_node + extra
        splits.append(weight_tensor[start:end, :])
        start = end

    return splits


def tensor_divide_by_neurons_ratios(weight_tensor, neuron_splits):
    """
    按指定神经元分割点分割全连接层权重

    参数:
        weight_tensor: [out_features, in_features]
        neuron_splits: [n1, n2, n3, ...] 各节点处理的神经元数

    返回:
        splits: List of weight tensors
    """
    out_features = weight_tensor.shape[0]

    if sum(neuron_splits) != out_features:
        raise ValueError(f"Neuron splits sum {sum(neuron_splits)} != out_features {out_features}")

    splits = []
    start = 0
    for n in neuron_splits:
        splits.append(weight_tensor[start:start + n, :])
        start += n

    return splits


def merge_neuron_outputs(outputs, activation_function=None):
    """
    合并神经元分割的输出
    """
    merged = torch.cat(outputs, dim=1)

    if activation_function:
        merged = activation_function(merged)

    return merged


def dispatch_featuremap_to_nodes(input_tensor, ratios):
    """
    将特征图按DDPG输出的比例分发到多个计算节点 (Filter Splitting)

    参数:
        input_tensor: [N, C_out, H, W] 特征图
        ratios: [r1, r2, r3, ...] 各节点分配比例，sum=1.0

    返回:
        splits: List of featuremap splits
    """
    C_out = input_tensor.shape[1]

    channel_splits = []
    for i, r in enumerate(ratios[:-1]):
        c = int(C_out * r)
        channel_splits.append(c)

    channel_splits.append(C_out - sum(channel_splits))

    return tensor_divide_by_filter_ratios(input_tensor, channel_splits, overlap=0)


def dispatch_fc_weights_to_nodes(fc_weight, ratios):
    """
    将全连接层权重按DDPG输出的比例分发到多个计算节点 (Neuron Splitting)

    参数:
        fc_weight: [out_features, in_features] 全连接层权重
        ratios: [r1, r2, r3, ...] 各节点分配比例，sum=1.0

    返回:
        splits: List of weight tensors
    """
    out_features = fc_weight.shape[0]

    neuron_splits = []
    for i, r in enumerate(ratios[:-1]):
        n = int(out_features * r)
        neuron_splits.append(n)

    neuron_splits.append(out_features - sum(neuron_splits))

    return tensor_divide_by_neurons_ratios(fc_weight, neuron_splits)


# ==================== VGG Layer Mapping for Filter Splitting ====================

VGG13_LAYER_MAPPING = {
    1: 'features_1_1',
    2: 'features_1_2',
    3: 'features_1_3',
    4: 'features_2_1',
    5: 'features_2_2',
    6: 'features_2_3',
    7: 'features_3_1',
    8: 'features_3_2',
    9: 'features_3_4',
    10: 'features_4_1',
    11: 'features_4_2',
    12: 'features_4_4',
    13: 'features_5_1',
    14: 'features_5_2',
    15: 'features_5_4',
    16: 'classifier_1',
    17: 'classifier_2',
    18: 'classifier_3',
}

VGG13_POOL_LAYERS = {3, 6, 9, 12, 15}

VGG13_CONV_LAYERS = {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}

VGG13_FC_LAYERS = {16, 17, 18}

VGG16_LAYER_MAPPING = {
    1: 'features_1_1',
    2: 'features_1_2',
    3: 'features_1_3',
    4: 'features_2_1',
    5: 'features_2_2',
    6: 'features_2_3',
    7: 'features_3_1',
    8: 'features_3_2',
    9: 'features_3_3',
    10: 'features_3_4',
    11: 'features_4_1',
    12: 'features_4_2',
    13: 'features_4_3',
    14: 'features_4_4',
    15: 'features_5_1',
    16: 'features_5_2',
    17: 'features_5_3',
    18: 'features_5_4',
    19: 'classifier_1',
    20: 'classifier_2',
    21: 'classifier_3',
}

VGG16_POOL_LAYERS = {3, 6, 10, 14, 18}

VGG16_CONV_LAYERS = {1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17}

VGG16_FC_LAYERS = {19, 20, 21}


class SlicedVGGExecutor:
    """
    AdapCP Filter Splitting 执行器

    使用 torch.nn.functional.conv2d 动态切片权重，
    支持在子节点上执行部分通道的卷积计算。
    """

    def __init__(self, full_model, model_type='VGG13'):
        """
        初始化 SlicedVGGExecutor

        参数:
            full_model: 完整的 VGG 模型
            model_type: 'VGG13' 或 'VGG16'
        """
        self.model = full_model
        self.model_type = model_type

        if model_type == 'VGG13':
            self.layer_mapping = VGG13_LAYER_MAPPING
            self.pool_layers = VGG13_POOL_LAYERS
            self.conv_layers = VGG13_CONV_LAYERS
            self.fc_layers = VGG13_FC_LAYERS
        elif model_type == 'VGG16':
            self.layer_mapping = VGG16_LAYER_MAPPING
            self.pool_layers = VGG16_POOL_LAYERS
            self.conv_layers = VGG16_CONV_LAYERS
            self.fc_layers = VGG16_FC_LAYERS
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.layer_modules = dict(full_model.named_modules())

    def get_layer_module(self, layer_id):
        """
        根据layer_id获取对应的模块名称

        参数:
            layer_id: 1-indexed layer ID

        返回:
            layer_name: 如 'features_1_1'
        """
        return self.layer_mapping.get(layer_id)

    def execute_sliced_conv(self, input_tensor, layer_id, start_filter, end_filter):
        """
        执行切片卷积 (使用F.conv2d动态切片权重)

        参数:
            input_tensor: [N, C_in, H, W] 输入特征图
            layer_id: 1-indexed layer ID
            start_filter: 起始通道索引
            end_filter: 结束通道索引

        返回:
            output_tensor: [N, end_filter-start_filter, H', W']
        """
        layer_name = self.get_layer_module(layer_id)
        if layer_name is None:
            raise ValueError(f"Invalid layer_id: {layer_id}")

        layer_module = self.layer_modules.get(layer_name)
        if layer_module is None:
            raise ValueError(f"Layer module not found: {layer_name}")

        # --- 新增修复：解包 Sequential 容器以寻找具有 weight 属性的真实网络层 ---
        if not hasattr(layer_module, 'weight'):
            for sub_module in layer_module.modules():
                if sub_module is not layer_module and hasattr(sub_module, 'weight'):
                    layer_module = sub_module
                    break
        # ----------------------------------------------------------------------

        if not hasattr(layer_module, 'weight'):
            raise ValueError(f"Layer {layer_name} does not have weight (not a conv/linear layer)")

        full_weight = layer_module.weight
        full_bias = layer_module.bias

        # 获取当前输入张量的通道数
        in_channels = input_tensor.size(1) 

        # --- 核心修复：同时切分输出维度(dim 0)和输入维度(dim 1) ---
        # 确保卷积核的输入通道维度与 input_tensor 匹配
        sliced_weight = full_weight[start_filter:end_filter, :in_channels, :, :]
        # -------------------------------------------------------

        if full_bias is not None:
            sliced_bias = full_bias[start_filter:end_filter]
        else:
            sliced_bias = None

        if hasattr(layer_module, 'stride'):
            stride = layer_module.stride
        else:
            stride = 1

        if hasattr(layer_module, 'padding'):
            padding = layer_module.padding
        else:
            padding = 0

        output_tensor = torch.nn.functional.conv2d(
            input_tensor, sliced_weight, sliced_bias,
            stride=stride, padding=padding
        )

        return output_tensor

    def execute_sliced_fc(self, input_tensor, layer_id, start_neuron, end_neuron):
        """
        执行切片全连接层 (ONS - Output Neuron Splitting)

        参数:
            input_tensor: [N, in_features] 输入
            layer_id: 1-indexed layer ID
            start_neuron: 起始神经元索引
            end_neuron: 结束神经元索引

        返回:
            output_tensor: [N, end_neuron-start_neuron]
        """
        layer_name = self.get_layer_module(layer_id)
        if layer_name is None:
            raise ValueError(f"Invalid layer_id: {layer_id}")

        layer_module = self.layer_modules.get(layer_name)
        if layer_module is None:
            raise ValueError(f"Layer module not found: {layer_name}")

        # --- 新增修复：解包 Sequential 容器以寻找具有 weight 属性的真实网络层 ---
        if not hasattr(layer_module, 'weight'):
            for sub_module in layer_module.modules():
                if sub_module is not layer_module and hasattr(sub_module, 'weight'):
                    layer_module = sub_module
                    break
        # ----------------------------------------------------------------------

        if not hasattr(layer_module, 'weight'):
            raise ValueError(f"Layer {layer_name} does not have weight (not a conv/linear layer)")

        full_weight = layer_module.weight
        full_bias = layer_module.bias

        sliced_weight = full_weight[start_neuron:end_neuron, :]

        if full_bias is not None:
            sliced_bias = full_bias[start_neuron:end_neuron]
        else:
            sliced_bias = None

        output_tensor = torch.nn.functional.linear(input_tensor, sliced_weight, sliced_bias)

        return output_tensor

    def execute_sliced_layer_range(self, input_tensor, start_layer, end_layer,
                                   start_filter, end_filter, current_channels=None):
        """
        执行连续多层切片卷积（用于单节点执行多层的情况）

        参数:
            input_tensor: [N, C_in, H, W] 输入特征图
            start_layer: 起始层ID (1-indexed)
            end_layer: 结束层ID (1-indexed)
            start_filter: 当前节点负责的起始通道
            end_filter: 当前节点负责的结束通道
            current_channels: 初始输入通道数（用于第一层之前的通道扩展）

        返回:
            output_tensor: 计算结果
        """
        current = input_tensor

        for layer_id in range(start_layer, end_layer + 1):
            if layer_id in self.pool_layers:
                current = torch.nn.functional.max_pool2d(current, kernel_size=2, stride=2)
                print(f"  Layer {layer_id}: MaxPool2d, output size: {current.size()}")

            elif layer_id in self.conv_layers:
                if layer_id == start_layer:
                    s_filter, e_filter = start_filter, end_filter
                else:
                    s_filter, e_filter = 0, current.size(1)

                current = self.execute_sliced_conv(current, layer_id, s_filter, e_filter)
                print(f"  Layer {layer_id}: Conv2d [{s_filter}:{e_filter}], output size: {current.size()}")

            elif layer_id in self.fc_layers:
                if layer_id == self.fc_layers and len(self.fc_layers) > 0:
                    min_fc = min(self.fc_layers)
                    if layer_id == min_fc:
                        current = torch.nn.functional.adaptive_avg_pool2d(current, (7, 7))
                        current = torch.flatten(current, 1)

                if layer_id == start_layer:
                    s_neuron, e_neuron = start_filter, end_filter
                else:
                    s_neuron, e_neuron = 0, current.size(1)

                current = self.execute_sliced_fc(current, layer_id, s_neuron, e_neuron)
                print(f"  Layer {layer_id}: Linear [{s_neuron}:{e_neuron}], output size: {current.size()}")

        return current

    def get_layer_info(self, layer_id):
        """
        获取指定层的权重信息

        返回:
            dict: {'name': str, 'weight_shape': tuple, 'bias_shape': tuple, 'type': 'conv'/'fc'/'pool'}
        """
        layer_name = self.get_layer_module(layer_id)
        if layer_name is None:
            return None

        layer_module = self.layer_modules.get(layer_name)
        if layer_module is None:
            return None

        if layer_id in self.pool_layers:
            return {'name': layer_name, 'type': 'pool'}

        if hasattr(layer_module, 'weight'):
            weight_shape = tuple(layer_module.weight.shape)
            bias_shape = tuple(layer_module.bias.shape) if layer_module.bias is not None else None
            layer_type = 'fc' if layer_id in self.fc_layers else 'conv'
            return {
                'name': layer_name,
                'weight_shape': weight_shape,
                'bias_shape': bias_shape,
                'type': layer_type
            }

        return None

