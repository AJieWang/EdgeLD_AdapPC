import sys
sys.path.append("../..")
sys.path.append("..")

import torch, time, socket, json, six
import torch.nn as nn
import numpy as np
from VGG.tensor_op import merge_total_tensor, merge_part_tensor, merge_total_tensor_pooled, merge_total_tensor_pabc

import threading

# IP设置
namenode_ip = "127.0.0.1"
# namenode_ip = "192.168.202.129"
datanode_ip = ["127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"]
# datanode_ip = ["192.168.202.130", "192.168.202.147", "192.168.202.131", "192.168.202.133", "192.168.202.134", "192.168.202.135"]
datanode_port = [10000, 10001, 10002, 10003, 10004, 10005]

namenode_pre_send = []
datanode_pre_send = []
after_receive = []
class Network_init_namenode():
    def __init__(self, namenode_num = 1, datanode_num = 1):
        super(Network_init_namenode, self).__init__()
        print ("NameNode 开始初始化")
        self.datanode_num = datanode_num
        self.client_socket = []
        if (datanode_num >= 1):
            for hostname in range(datanode_num):
                tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                tcp_client_socket.connect((datanode_ip[hostname], datanode_port[hostname]))
                hello_world = "Hello DataNode "+ str(hostname) + ", I'm NameNode"
                tcp_client_socket.send(hello_world.encode())

                recv_data_test = tcp_client_socket.recv(1024)
                print (str(recv_data_test, encoding="UTF-8"))
                self.client_socket.append(tcp_client_socket)
        print ("NameNode 初始化完成")
        self.recv_tensor_temp_list = []
        for it in range(datanode_num):
            self.recv_tensor_temp_list.append(0)

    def get_recv_tensor_list(self):
        return self.recv_tensor_temp_list

    def get_merged_total_tensor(self, divide_record = 0, cross_layer = 1):
        temp = merge_total_tensor(self.recv_tensor_temp_list, divide_record = divide_record, cross_layer = cross_layer)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp

    # ===================== 新增优化：merge_total_tensor_pooled =====================
    def get_merged_total_tensor_pooled(self, divide_record = 0, cross_layer = 1):
        temp = merge_total_tensor_pooled(self.recv_tensor_temp_list, divide_record = divide_record, cross_layer = cross_layer)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp
    # ==============================================================================

    # ===================== 新增优化：merge_total_tensor_pabc =====================
    def get_merged_total_tensor_pabc(self, divide_record = 0, cross_layer = 1):
        temp = merge_total_tensor_pabc(self.recv_tensor_temp_list, divide_record = divide_record, cross_layer = cross_layer)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp
    # ==============================================================================

    def get_merged_part_tensor(self):
        temp = merge_part_tensor(self.recv_tensor_temp_list, divide_record=0, cross_layer=1)
        # # 原先的recv_tensor_temp_list重新初始化
        # for i in range(self.datanode_num):
        #     self.recv_tensor_temp_list[i] = torch.rand(1, 1, 1, 1)
        return temp

    def namenode_send_data(self, datanode_name, input_tensor, start, end, transfer_time):
        pre_send_time = time.time()
        # 先发送数据长度，再发送数据
        input_numpy = input_tensor.detach().numpy()
        start = str(start).encode(encoding='utf-8')
        end = str(end).encode(encoding='utf-8')
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()
        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        # 发送长度
        send_data_len = str(len(send_data)).encode(encoding='utf-8')
        # print("send_data长度：", len(send_data))
        self.client_socket[datanode_name].send(send_data_len)
        time.sleep(0.01)
        # 发送数据

        transfer_start_time = time.time()
        temp_time = transfer_start_time - pre_send_time
        namenode_pre_send.append(temp_time)
        print('NameNode Pre send time: %0.3fs, Total pre send time: %0.3fs' % (temp_time, sum(namenode_pre_send)))

        self.client_socket[datanode_name].sendall(send_data)

        transfer_time.append(time.time() - transfer_start_time)

        # print("namenode socket 数据发送完成")
        # print("send_return_info: ", send_return_info)

        # 发送数据后等待接收datanode返回的数据
        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp
        # print("最终接受数据的长度：", len(recv_data), recv_data_len)

        after_receive_start_time = time.time()
        split_list = recv_data.split(b'@#$%')
        recv_start = int(str(split_list[0], encoding="UTF-8"))
        recv_end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        self.recv_tensor_temp_list[datanode_name] = recv_tensor

        temp_time = time.time() - after_receive_start_time
        after_receive.append(temp_time)
        print('NameNode After receive time: %0.3fs, Total after receive time: %0.3fs' % (temp_time, sum(after_receive)))
        return recv_start, recv_end, recv_tensor

    def namenode_recv_data(self, datanode_name):
        # 先接收数据长度，再接收数据
        # 接收的数据：start 0，end 0， numpy_size, tensor
        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding='utf-8'))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp
        # print("最终接受数据的长度：", len(recv_data), recv_data_len)

        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        return start, end, recv_tensor

    def close(self, datanode_name):
        self.client_socket[datanode_name].close()
    def close_all(self):
        for i in range(self.datanode_num):
            self.client_socket[i].close()

class Network_init_datanode():
    def __init__(self, namenode_num = 1, datanode_num = 3, datanode_name = 0):
        super(Network_init_datanode, self).__init__()
        print ("DataNode %d 开始初始化" % datanode_name )
        # 创建服务器 socket
        tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_server_socket.bind((datanode_ip[datanode_name], datanode_port[datanode_name]))
        tcp_server_socket.listen(2)
        # 得到client socket
        self.datanode_socket, client_addr = tcp_server_socket.accept()
        recv_data_test = self.datanode_socket.recv(1024)
        # 简单的测试
        print (str(recv_data_test, encoding="UTF-8"))
        # 发送数据测试
        hello_world = "Hello NameNode, I have received your hello world, I'm DateNode " + str(datanode_name)
        self.datanode_socket.send(hello_world.encode())
        print("DataNode %d 初始化完成\n" % datanode_name)

        # 初始化一些参数，中间计算结果分割为saved_tensor和divied_tensor，saved_tensor保存在本机，divied_tensor发送至namenode
        self.datanode_num = datanode_num
        self.datanode_name = datanode_name
        self.last_inference_layer = 0
        self.saved_tensor = torch.rand(1, 1, 1, 1) # tensor数据保存在本机
        self.divied_tensor_list = [] # 数据格式为list，发送至namenode
        # 最左侧 或 最右侧, 数据初始化
        if datanode_name==0 or datanode_name == datanode_num-1:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
        # 中间的数据，左右两侧都会拆除，数据初始化
        else:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
    # set, get 函数
    def set_last_inference_layer(self, layer):
        self.last_inference_layer = layer
    def set_saved_tensor(self, tensor):
        self.saved_tensor = tensor
    def set_divied_tensor_list(self, tensor_list):
        self.divied_tensor_list = tensor_list
    def get_last_inference_layer(self):
        return int(self.last_inference_layer)
    def get_saved_tensor(self):
        return self.saved_tensor
    # 根据 datanode_name 的不同，将一个或两个divied_tensor
    def get_divied_merged_tensor(self):
        if self.datanode_name == 0 or self.datanode_num - 1:
            return self.divied_tensor_list[0]
        else:
            return torch.cat((self.divied_tensor_list[0], self.divied_tensor_list[1]), 3)
    # 根据 datanode_name 的不同，合并 saved_tensor 和 divied_tensor_list
    def get_merged_tensor(self):

        if self.datanode_name == 0:
            merged_tensor = torch.cat((self.saved_tensor, self.divied_tensor_list[0]), 3)
        elif self.datanode_name == self.datanode_num - 1:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor), 3)
        else:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor, self.divied_tensor_list[1]), 3)
        return merged_tensor
    # 根据要求清空saved_tensor 和 divied_tensor_list
    def empty_tensor(self):
        self.saved_tensor = torch.rand(1, 1, 1, 1)
        if self.datanode_name==0 or self.datanode_name == self.datanode_num-1:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
        # 中间的数据，左右两侧都会拆除，数据初始化
        else:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
            self.divied_tensor_list[1] = torch.rand(1, 1, 1, 1)


    def datanode_send_data(self, input_tensor, transfer_time, start=0, end=0):
        # 先发送数据长度，再发送数据
        # print ("datanode_socket 数据发送开始")
        pre_send_time = time.time()
        input_numpy = input_tensor.detach().numpy()

        start = str(start).encode(encoding="UTF-8")
        end = str(end).encode(encoding="UTF-8")
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()

        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        # 发送长度
        send_data_len = str(len(send_data)).encode(encoding="UTF-8")
        # print("send_data长度：", len(send_data))
        self.datanode_socket.send(send_data_len)
        time.sleep(0.01)
        # 发送数据
        temp_time = time.time() - pre_send_time
        datanode_pre_send.append(temp_time)
        print('DataNode Pre send time: %0.3fs, Total pre send time: %0.3fs' % (temp_time, sum(datanode_pre_send)))

        transfer_start_time = time.time()

        self.datanode_socket.sendall(send_data)

        transfer_time.append(time.time() - transfer_start_time)

        # print ("datanode_socket 数据发送完成")

    def datanode_recv_data(self, pre_conv):
        # 先接受长度，再接收数据。
        data_total_len = b''
        while True:
            data = self.datanode_socket.recv(1024)
            if len(data) != 0:
                data_total_len = data
                break
        pre_conv_start_time = time.time()

        print ("DataNode recv data length: ", str(data_total_len, encoding='utf-8'))
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        print("DataNode recv data length: ", data_total_len)
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.datanode_socket.recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        # print ("最终接受数据的长度：", len(recv_data), recv_data_len)
        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        # print ("recv_numpy_size: ", recv_numpy_size)
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)

        pre_conv.append(time.time() - pre_conv_start_time)
        return start, end, recv_tensor

    def close(self):
        self.datanode_socket.close()

def get_recv_tensor_size(split_list_bytes):
    split_str = str(split_list_bytes, encoding="UTF-8").split("*")
    # print ("split_str:", split_str)
    recv_numpy_size = []
    for i ,value in enumerate(split_str):
        recv_numpy_size.append(int(value))
    # print ("recv_numpy_size:", type(tuple(recv_numpy_size)))
    return tuple(recv_numpy_size)

def get_numpy_size(input_tensor):

    size_list = list(input_tensor.size())
    input_numpy_size = ""
    length = len(size_list)
    for i, value in enumerate(size_list):
        if i == length - 1:
            input_numpy_size += str(value)
        else:
            input_numpy_size += str(value)
            input_numpy_size += "*"
    return input_numpy_size.encode(encoding="UTF-8")



# 之前的备份
# class Network_init_namenode():
#     def __init__(self, namenode_num = 1, datanode_num = 1):
#         super(Network_init_namenode, self).__init__()
#         print ("NameNode 开始初始化")
#         self.datanode_num = datanode_num
#         self.client_socket = []
#         if (datanode_num >= 1):
#             for hostname in range(datanode_num):
#                 tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 tcp_client_socket.connect((datanode_ip[hostname], datanode_port[hostname]))
#                 hello_world = "Hello DataNode "+ str(hostname) + ", I'm NameNode"
#                 tcp_client_socket.send(hello_world.encode())
#
#                 recv_data_test = tcp_client_socket.recv(1024)
#                 print (str(recv_data_test, encoding="UTF-8"))
#                 self.client_socket.append(tcp_client_socket)
#         print ("NameNode 初始化完成")
#
#     def namenode_send_data(self, i, input_tensor, start, end):
#         # 先发送数据长度，再发送数据
#         input_numpy = input_tensor.detach().numpy()
#         start = str(start).encode(encoding='utf-8')
#         end = str(end).encode(encoding='utf-8')
#         input_numpy_size = get_numpy_size(input_tensor)
#         input_bytes = input_numpy.tostring()
#         send_data = start + b':::' + end + b':::' + input_numpy_size + b':::' + input_bytes
#         # 发送长度
#         send_data_len = str(len(send_data)).encode(encoding='utf-8')
#         # print("send_data长度：", len(send_data))
#         self.client_socket[i].send(send_data_len)
#         # 发送数据
#         self.client_socket[i].sendall(send_data)
#         # print("namenode socket 数据发送完成")
#         # print("send_return_info: ", send_return_info)
#
#     def namenode_recv_data(self, i):
#         # 先接收数据长度，再接收数据
#         # 接收的数据：start 0，end 0， numpy_size, tensor
#         data_total_len = self.client_socket[i].recv(1024)
#         data_total_len = int(str(data_total_len, encoding='utf-8'))
#         recv_data_len = 0
#         recv_data = b''
#         while recv_data_len < data_total_len:
#             recv_data_temp = self.client_socket[i].recv(10240)
#             recv_data_len += len(recv_data_temp)
#             recv_data += recv_data_temp
#         # print("最终接受数据的长度：", len(recv_data), recv_data_len)
#
#         split_list = recv_data.split(b':::')
#         start = int(str(split_list[0], encoding="utf-8"))
#         end = int(str(split_list[1], encoding="utf-8"))
#
#         recv_numpy_size = get_recv_tensor_size(split_list[2])
#         # print("recv_numpy_size: ", recv_numpy_size)
#         recv_numpy = np.fromstring(split_list[3], dtype=np.float32)
#         recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
#         recv_tensor = torch.from_numpy(recv_numpy)
#         return start, end, recv_tensor
#
#     def close(self, i):
#         self.client_socket[i].close()
#     def close_all(self):
#         for i in range(self.datanode_num):
#             self.client_socket[i].close()
#
# class Network_init_datanode():
#     def __init__(self, namenode_num = 1, datanode_num = 3, hostname = 0):
#         super(Network_init_datanode, self).__init__()
#         print ("DataNode %d 开始初始化" % hostname )
#         # 创建服务器 socket
#         tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         tcp_server_socket.bind((datanode_ip[hostname], datanode_port[hostname]))
#         tcp_server_socket.listen(2)
#         # 得到client socket
#         self.datanode_socket, client_addr = tcp_server_socket.accept()
#         recv_data_test = self.datanode_socket.recv(1024)
#         # 简单的测试
#         print (str(recv_data_test, encoding="UTF-8"))
#         # 发送数据测试
#         hello_world = "Hello NameNode, I have received your hello world, I'm DateNode " + str(hostname)
#         self.datanode_socket.send(hello_world.encode())
#         print("DataNode %d 初始化完成" % hostname)
#
#     def datanode_send_data(self, input_tensor, start=0, end=0):
#         # 先发送数据长度，再发送数据
#         # print ("datanode_socket 数据发送开始")
#         input_numpy = input_tensor.detach().numpy()
#
#         start = str(start).encode(encoding='utf-8')
#         end = str(end).encode(encoding='utf-8')
#         input_numpy_size = get_numpy_size(input_tensor)
#         input_bytes = input_numpy.tostring()
#
#         send_data = start + b':::' + end + b':::' + input_numpy_size + b':::' + input_bytes
#         # 发送长度
#         send_data_len = str(len(send_data)).encode(encoding='utf-8')
#         # print("send_data长度：", len(send_data))
#         self.datanode_socket.send(send_data_len)
#         # 发送数据
#         self.datanode_socket.sendall(send_data)
#         # print ("datanode_socket 数据发送完成")
#
#     def datanode_recv_data(self):
#         # 先接受长度，再接收数据。
#         data_total_len = b''
#         while True:
#             data = self.datanode_socket.recv(1024)
#             if len(data) != 0:
#                 data_total_len = data
#                 break
#         # print ("DataNode recv data length: ", str(data_total_len, encoding='utf-8'))
#         data_total_len = int(str(data_total_len, encoding='utf-8'))
#         print("DataNode recv data length: ", data_total_len)
#         recv_data_len = 0
#         recv_data = b''
#         while recv_data_len < data_total_len:
#             recv_data_temp = self.datanode_socket.recv(10240)
#             recv_data_len += len(recv_data_temp)
#             recv_data += recv_data_temp
#
#         # print ("最终接受数据的长度：", len(recv_data), recv_data_len)
#         split_list = recv_data.split(b':::')
#         start = int(str(split_list[0], encoding="utf-8"))
#         end = int(str(split_list[1], encoding="utf-8"))
#
#         recv_numpy_size = get_recv_tensor_size(split_list[2])
#         # print ("recv_numpy_size: ", recv_numpy_size)
#         recv_numpy = np.fromstring(split_list[3], dtype = np.float32)
#         recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
#         recv_tensor = torch.from_numpy(recv_numpy)
#         return start, end, recv_tensor
#
#     def close(self):
#         self.datanode_socket.close()
#
# def get_recv_tensor_size(split_list_bytes):
#     split_str = str(split_list_bytes, encoding='utf-8').split("*")
#     # print ("split_str:", split_str)
#     recv_numpy_size = []
#     for i ,value in enumerate(split_str):
#         recv_numpy_size.append(int(value))
#     # print ("recv_numpy_size:", type(tuple(recv_numpy_size)))
#     return tuple(recv_numpy_size)
#
# def get_numpy_size(input_tensor):
#
#     size_list = list(input_tensor.size())
#     input_numpy_size = ""
#     length = len(size_list)
#     for i, value in enumerate(size_list):
#         if i == length - 1:
#             input_numpy_size += str(value)
#         else:
#             input_numpy_size += str(value)
#             input_numpy_size += "*"
#     return input_numpy_size.encode(encoding="utf-8")


class EdgeNNNode:
    """
    边缘协作网络通信类（替代AllReduce）

    支持：
    1. 广播plan和初始输入给所有DataNode
    2. 收集各DataNode的部分通道输出
    3. 广播合并后的完整张量给所有DataNode
    """

    def __init__(self, namenode):
        self.namenode = namenode
        self.datanode_num = namenode.datanode_num

    def broadcast_plan_to_all(self, plan, boundaries, input_tensor, c_out_list):
        """
        广播执行计划给所有DataNode

        协议: start_layer@#$%end_layer@#$%n_boundaries@#$%boundary0@#$%boundary1@#$%...@#$%n_c_out@#$%c_out0@#$%c_out1@#$%...@#$%tensor

        参数:
            plan: dict, {'start_layer': int, 'end_layer': int}
            boundaries: list, 通道分割比例边界，如 [0, 0.5, 0.8, 1.0]
            input_tensor: 完整的输入张量
            c_out_list: list, 各层的输出通道数
        """
        for dn_id in range(self.datanode_num):
            self._send_broadcast_task(dn_id, plan['start_layer'], plan['end_layer'], boundaries, input_tensor, c_out_list)

    def _send_broadcast_task(self, datanode_id, start_layer, end_layer, boundaries, input_tensor, c_out_list):
        start_str = str(start_layer).encode('utf-8')
        end_str = str(end_layer).encode('utf-8')
        n_boundaries = str(len(boundaries)).encode('utf-8')

        input_numpy = input_tensor.detach().numpy()
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()

        boundaries_bytes = b'%&%'.join([str(b).encode('utf-8') for b in boundaries])

        n_c_out = str(len(c_out_list)).encode('utf-8')
        c_out_bytes = b'%&%'.join([str(c).encode('utf-8') for c in c_out_list])

        send_data = (start_str + b'@#$%' + end_str + b'@#$%' + n_boundaries + b'@#$%' + 
                     boundaries_bytes + b'@#$%' + n_c_out + b'@#$%' + c_out_bytes + b'@#$%' + 
                     input_numpy_size + b'@#$%' + input_bytes)

        send_data_len = str(len(send_data)).encode('utf-8').ljust(16)
        self.namenode.client_socket[datanode_id].sendall(send_data_len)
        self.namenode.client_socket[datanode_id].sendall(send_data)

        print(f"[Broadcast] To Node {datanode_id}: layers {start_layer}-{end_layer}, boundaries {boundaries}")

    def collect_slice_from_datanode(self, datanode_id, transfer_time):
        """
        从DataNode收集切片结果

        协议: layer_id@#$%node_id@#$%tensor

        返回:
            tuple: (layer_id, node_id, tensor_slice)
        """
        data_total_len_bytes = b''
        while len(data_total_len_bytes) < 16:
            packet = self.namenode.client_socket[datanode_id].recv(16 - len(data_total_len_bytes))
            if not packet:
                break
            data_total_len_bytes += packet

        data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.namenode.client_socket[datanode_id].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')

        layer_id = int(str(split_list[0], encoding='utf-8'))
        node_id = int(str(split_list[1], encoding='utf-8'))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        recv_numpy = np.frombuffer(split_list[3], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        tensor_slice = torch.from_numpy(recv_numpy)

        transfer_time.append(sum(transfer_time) if transfer_time else 0)

        print(f"[Collect] From Node {node_id}, Layer {layer_id}: {tensor_slice.size()}")

        return layer_id, node_id, tensor_slice

    def broadcast_merged_to_all(self, merged_tensor, next_layer_id=None):
        """
        广播合并后的完整张量给所有DataNode

        协议: next_layer_id@#$%tensor (next_layer_id为0表示无下一层)

        参数:
            merged_tensor: 合并后的完整张量
            next_layer_id: 下一层的layer_id，0表示推理结束
        """
        for dn_id in range(self.datanode_num):
            self._send_merged_task(dn_id, merged_tensor, next_layer_id)

    def _send_merged_task(self, datanode_id, merged_tensor, next_layer_id):
        next_layer_str = str(next_layer_id).encode('utf-8')

        input_numpy = merged_tensor.detach().numpy()
        input_numpy_size = get_numpy_size(merged_tensor)
        input_bytes = input_numpy.tobytes()

        send_data = next_layer_str + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes

        send_data_len = str(len(send_data)).encode('utf-8').ljust(16)
        self.namenode.client_socket[datanode_id].sendall(send_data_len)
        self.namenode.client_socket[datanode_id].sendall(send_data)

        print(f"[BroadcastMerged] To Node {datanode_id}: next_layer={next_layer_id}, size={merged_tensor.size()}")


class EdgeDataNode:
    """
    边缘协作子节点网络通信类（替代AllReduceDataNode）

    支持：
    1. 接收主节点广播的plan和初始输入
    2. 执行切片计算后提交切片给主节点
    3. 接收主节点广播的合并后完整张量
    """

    def __init__(self, datanode):
        self.datanode = datanode
        self.datanode_name = datanode.datanode_name

    def receive_initial_broadcast(self):
        """
        接收主节点广播的plan和初始输入

        协议: start@#$%end@#$%n_boundaries@#$%boundary0@#$%boundary1@#$%...@#$%n_c_out@#$%c_out0@#$%c_out1@#$%...@#$%tensor

        返回:
            dict: {'start_layer': int, 'end_layer': int, 'filter_boundaries': list, 'c_out_list': list, 'input_tensor': Tensor}
        """
        data_total_len_bytes = b''
        while len(data_total_len_bytes) < 16:
            packet = self.datanode.datanode_socket.recv(16 - len(data_total_len_bytes))
            if not packet:
                break
            data_total_len_bytes += packet

        data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.datanode.datanode_socket.recv(min(10240, data_total_len - recv_data_len))
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')

        start_layer = int(str(split_list[0], encoding='utf-8'))
        end_layer = int(str(split_list[1], encoding='utf-8'))
        n_boundaries = int(str(split_list[2], encoding='utf-8'))

        boundaries_part = split_list[3]
        boundaries_list = []
        for b in boundaries_part.split(b'%&%'):
            if b:
                boundaries_list.append(float(str(b, encoding='utf-8')))

        n_c_out = int(str(split_list[4], encoding='utf-8'))
        
        # 获取用 %&% 拼接的 c_out 字符串，并进行二次拆分
        c_out_part = split_list[5]
        c_out_list = []
        for c in c_out_part.split(b'%&%'):
            if c:
                c_out_list.append(int(str(c, encoding='utf-8')))

        # 因为 c_out_bytes 是一个整体占据了索引 5，所以 numpy 数据的索引是固定的 6 和 7
        recv_numpy_size = get_recv_tensor_size(split_list[6])
        recv_numpy = np.frombuffer(split_list[7], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        input_tensor = torch.from_numpy(recv_numpy)

        print(f"[ReceiveInitial] From Master: layers {start_layer}-{end_layer}, boundaries {boundaries_list}, c_out_list {c_out_list}, input {input_tensor.size()}")

        return {
            'start_layer': start_layer,
            'end_layer': end_layer,
            'filter_boundaries': boundaries_list,
            'c_out_list': c_out_list,
            'input_tensor': input_tensor
        }

    def send_slice_to_master(self, layer_id, tensor_slice):
        """
        向主节点发送切片结果

        协议: layer_id@#$%node_id@#$%tensor

        参数:
            layer_id: 当前层ID
            tensor_slice: 计算得到的切片张量
        """
        layer_str = str(layer_id).encode('utf-8')
        node_str = str(self.datanode_name).encode('utf-8')

        if tensor_slice is None:
            tensor_slice = torch.zeros(1, 0, 1, 1)

        input_numpy = tensor_slice.detach().numpy()
        input_numpy_size = get_numpy_size(tensor_slice)
        input_bytes = input_numpy.tobytes()

        send_data = layer_str + b'@#$%' + node_str + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes

        send_data_len = str(len(send_data)).encode('utf-8').ljust(16)
        self.datanode.datanode_socket.sendall(send_data_len)
        self.datanode.datanode_socket.sendall(send_data)

        print(f"[SendSlice] To Master: Layer {layer_id}, Node {self.datanode_name}, size {tensor_slice.size()}")

    def receive_merged_tensor(self):
        """
        接收主节点广播的合并后完整张量

        协议: next_layer_id@#$%tensor

        返回:
            dict: {'next_layer_id': int, 'tensor': Tensor}
            如果 next_layer_id == 0 表示推理结束
        """
        data_total_len_bytes = b''
        while len(data_total_len_bytes) < 16:
            packet = self.datanode.datanode_socket.recv(16 - len(data_total_len_bytes))
            if not packet:
                break
            data_total_len_bytes += packet

        data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.datanode.datanode_socket.recv(min(10240, data_total_len - recv_data_len))
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')

        next_layer_id = int(str(split_list[0], encoding='utf-8'))

        recv_numpy_size = get_recv_tensor_size(split_list[1])
        recv_numpy = np.frombuffer(split_list[2], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        tensor = torch.from_numpy(recv_numpy)

        print(f"[ReceiveMerged] From Master: next_layer={next_layer_id}, size {tensor.size()}")

        return {
            'next_layer_id': next_layer_id,
            'tensor': tensor
        }

    def close(self):
        self.datanode.close()

# ---------------------- 增加 P2P 通信类 ----------------------
class EdgeP2PCommunicator:
    """
    DataNode 之间的边边通信类 (P2P Mesh Network)
    用于在连续卷积层计算间，互相交换并合并分割后的特征图 (Filter Splitting 合并)
    """
    def __init__(self, datanode_name, total_datanodes, ip_list, base_port=20000):
        self.node_id = datanode_name
        self.total_nodes = total_datanodes
        self.ip_list = ip_list
        self.base_port = base_port
        self.sockets = {}  # 保存与其他 DataNode 的连接 {node_id: socket}

    def initialize_p2p_network(self):
        print(f"[P2P] DataNode {self.node_id} 开始建立边边通信网络...")
        
        # 1. 作为服务端，监听来自 node_id 更大的节点的连接
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.ip_list[self.node_id], self.base_port + self.node_id))
        self.server_sock.listen(self.total_nodes)

        # 2. 作为客户端，主动连接 node_id 更小的节点
        for target_id in range(self.node_id):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while True:
                try:
                    sock.connect((self.ip_list[target_id], self.base_port + target_id))
                    # 发送自己的 node_id 作为身份验证
                    sock.sendall(str(self.node_id).encode('utf-8').ljust(16))
                    self.sockets[target_id] = sock
                    print(f"[P2P] DataNode {self.node_id} 已连接到 DataNode {target_id}")
                    break
                except ConnectionRefusedError:
                    time.sleep(0.5)

        # 3. 作为服务端，接受 node_id 更大的节点的连接
        for _ in range(self.node_id + 1, self.total_nodes):
            conn, addr = self.server_sock.accept()
            # 接收对方的身份
            remote_id = int(conn.recv(16).decode('utf-8').strip())
            self.sockets[remote_id] = conn
            print(f"[P2P] DataNode {self.node_id} 接受了 DataNode {remote_id} 的连接")

        print(f"[P2P] DataNode {self.node_id} 边边通信网络建立完成！")

    def all_gather_tensor(self, my_tensor_slice):
        """
        核心操作：发送自己的张量切片给所有人，接收所有人的切片，按通道拼接成完整的张量
        """
        gathered_slices = {self.node_id: my_tensor_slice}

        # 为了避免死锁，使用多线程同时接收其他节点的数据
        def recv_task(remote_id, sock):
            tensor = self._recv_tensor(sock)
            gathered_slices[remote_id] = tensor

        threads = []
        for remote_id, sock in self.sockets.items():
            t = threading.Thread(target=recv_task, args=(remote_id, sock))
            t.start()
            threads.append(t)

        # 主线程负责向所有节点发送自己的数据
        for remote_id, sock in self.sockets.items():
            self._send_tensor(sock, my_tensor_slice)

        # 等待所有接收任务完成
        for t in threads:
            t.join()

        # 按照 node_id 的顺序将特征图在 通道维度(dim=1) 上拼接
        sorted_tensors = [gathered_slices[i] for i in range(self.total_nodes)]
        merged_tensor = torch.cat(sorted_tensors, dim=1)
        
        return merged_tensor

    def _send_tensor(self, sock, tensor):
        input_numpy = tensor.detach().numpy()
        input_numpy_size = get_numpy_size(tensor)
        input_bytes = input_numpy.tobytes()
        send_data = input_numpy_size + b'@#$%' + input_bytes
        send_data_len = str(len(send_data)).encode('utf-8').ljust(16)
        sock.sendall(send_data_len)
        sock.sendall(send_data)

    def _recv_tensor(self, sock):
        data_total_len_bytes = b''
        while len(data_total_len_bytes) < 16:
            packet = sock.recv(16 - len(data_total_len_bytes))
            if not packet:
                break
            data_total_len_bytes += packet
        data_total_len = int(data_total_len_bytes.decode('utf-8').strip())

        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = sock.recv(min(10240, data_total_len - recv_data_len))
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')
        recv_numpy_size = get_recv_tensor_size(split_list[0])
        recv_numpy = np.frombuffer(split_list[1], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        return torch.from_numpy(recv_numpy)

    def close_all(self):
        for sock in self.sockets.values():
            sock.close()
        self.server_sock.close()
# ---------------------- 增加 P2P 通信类 ----------------------