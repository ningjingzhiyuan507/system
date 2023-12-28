import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
use_cuda = torch.cuda.is_available()


class IO_B(nn.Module):
    """docstring for IO_B: 定义自己的网络，继承nn.Module,实现forward方法"""

    def __init__(self, dim, hid):
        """定义一系列输入层、隐藏层和输出层"""
        super(IO_B, self).__init__()
        self.dim = dim  # 每个节点的信息，如spring每个节点位置和速度各有两个维度信息，就是4
        self.hid = hid  # 隐藏层节点数量
        self.n2e = nn.Linear(2 * dim, hid)  # 设置网络中的全连接层（线性层）
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)

    def forward(self, x, adj):
        """
        实现各层之间的连接关系（实质就是前向传播过程）
        :param x: features of all nodes at time t,[b*n*d]
        :param adj: i th column of adj mat,[n*1]
        :return:前向传播输出层的结果(预测值y_hat)
        """
        out = x[:, :, :]
        innode = out.unsqueeze(1).repeat(1, adj.size()[1], 1, 1)
        outnode = innode.transpose(1, 2)
        node2edge = F.relu(self.n2e(torch.cat((innode, outnode), 3)))
        edge2edge = F.relu(self.e2e(node2edge))

        adjs = adj.view(adj.size()[0], adj.size()[1], 1)
        adjs = adjs.repeat(1, 1, edge2edge.size()[3])
        edges = adjs * edge2edge

        out = torch.sum(edges, 1)
        out = F.relu(self.e2n(out))
        out = F.relu(self.n2n(out))
        out = torch.cat((x, out), dim=-1)
        out = self.output(out)

        return out


        # starter = x[:, i, :]
        # x_total_sum = 0
        #
        # current_x = x[:, :, :]
        # current_adj_col = adj_col[:]
        # ender = x[:, i, :]
        # ender = ender.unsqueeze(1)
        # ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))
        # c_x = torch.cat((current_x, ender), 2)  # 1,6,2
        #
        # c_x = F.relu(self.n2e(c_x))
        # c_x = F.relu(self.e2e(c_x))
        #
        # c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)
        # current_x_sum = torch.sum(c_x, 1)
        # x_total_sum = x_total_sum + current_x_sum
        #
        # x = F.relu(self.e2n(x_total_sum))
        # x = F.relu(self.n2n(x))
        #
        # x = torch.cat((starter, x), dim=-1)
        # x = self.output(x)
        # return x


class IO_B_Voter(nn.Module):
    """docstring for IO_B"""

    def __init__(self, dim, hid):
        super(IO_B_Voter, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, adj_col, i, num, node_size):
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        #num =node_num//node_size ,node_num is the total number of nodes
        #node_size: In order to save memory, the information of node i is only
        # combined with the information of node_size nodes at a time
        #eg.We have a total of 2000 node information,
        # and setting node_size to 800 means that the i-th node only ocombined with the information of 800 nodes at a time at a time.
        # At this time, num=2000//800=2
        starter = x[:, i, :]
        x_total_sum = 0
        for n in range(num + 1):
            if n != num:
                current_x = x[:, n * node_size:(n + 1) * node_size, :]
                current_adj_col = adj_col[n * node_size:(n + 1) * node_size]
            else:
                current_x = x[:, n * node_size:, :]
                current_adj_col = adj_col[n * node_size:]
            ender = x[:, i, :]
            ender = ender.unsqueeze(1)
            ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))
            c_x = torch.cat((current_x, ender), 2)

            c_x = F.relu(self.n2e(c_x))
            c_x = F.relu(self.e2e(c_x))

            c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)
            current_x_sum = torch.sum(c_x, 1)
            x_total_sum = x_total_sum + current_x_sum

        x = F.relu(self.e2n(x_total_sum))
        x = F.relu(self.n2n(x))

        x = torch.cat((starter, x), dim=-1)
        x = self.output(x)
        x = self.logsoftmax(x)  # 对预测值进行log softmax转换为概率值

        return x

#####################
# Network Generator #
#####################


class Gumbel_Generator_Old(nn.Module):
    """Gumbel分布网络生成器"""
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_Old, self).__init__()
        self.sz = sz
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    # output: a matrix
    def sample_all(self, hard=False, epoch=1):
        """采样得到一个邻接矩阵"""
        self.logp = self.gen_matrix.view(-1, 2)
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        if use_cuda:
            out = out.cuda()
        out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        # 1000 50
        # if epoch > 998:
        #     for i in range(out_matrix.size()[0]):
        #         for j in range(out_matrix.size()[1]):
        #             if out_matrix[i][j].item() == 1:
        #                 out_matrix[j][i] = 1
        return out_matrix

    # output: the i-th column of matrix
    def sample_adj_i(self, i, hard=False, sample_time=1):
        """
        网络生成器模型根据当前数据对第i个节点的预测值
        :param i:
        :param hard:
        :param sample_time:
        :return:
        """
        self.logp = self.gen_matrix[:, i]
        out = gumbel_softmax(self.logp, self.temperature, hard=hard)
        if use_cuda:
            out = out.cuda()
        if hard:
            out_matrix = out.float()
        else:
            out_matrix = out[:, 0]
        return out_matrix

    def get_temperature(self):
        return self.temperature

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)


class Controller(nn.Module):

    def __init__(self, dim, hid):
        super(Controller, self).__init__()
        self.dim = dim
        self.hid = hid
        self.linear1 = nn.Linear(self.dim, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.linear3 = nn.Linear(self.hid, self.hid)
        self.linear4 = nn.Linear(self.hid, self.hid)
        self.output = nn.Linear(self.hid+self.dim, self.dim//2)

    def forward(self, cur_x, diff_x, adj_col, except_control_node_list):
        """
        controller前向传播
        :param cur_x: (128,4)驱动点k在128个时间点上的loc_vel
        :param diff_x: (128,7,4)被动点在128个时间点上的loc_vel
        :param adj_col: (10)驱动点k与其它节点的邻接概率向量
        :param except_control_node_list: (7)被动点ID
        :return: 驱动点在128个时间点的速度（要想达到目标，每个驱动点在对应时间点应该有的速度）
        """
        starter = diff_x  # 128,7,4

        x = F.relu(self.linear1(starter))  # 128,7,128
        x = F.relu(self.linear2(x))  # 128,7,128
        adj_col = adj_col[except_control_node_list]  # 驱动点k与7个被动点的邻接概率向量
        x = x * adj_col.unsqueeze(1).expand(adj_col.size(0), self.hid)  # 128,7,128

        x_sum = torch.sum(x, 1)  # 128,128

        x = F.relu(self.linear3(x_sum))  # 128,128
        x = F.relu(self.linear4(x))  # 128,128

        x = torch.cat((cur_x, x), dim=-1)  # 128,128+4
        x  = self.output(x)  # 128,2
        #x = torch.sigmoid(x) # if dyn is CMN

        return x


'''network completetion'''
class Gumbel_Generator_nc(nn.Module):
    """Gumbel分布网络生成器"""
    def __init__(self, sz=10, del_num=1, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_nc, self).__init__()
        self.sz = sz
        self.del_num = del_num
        self.gen_matrix = Parameter(
            torch.rand(del_num * (2 * sz - del_num - 1) // 2, 2))  # cmy get only unknown part parameter
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False):
        self.logp = self.gen_matrix
        if use_cuda:
            self.logp = self.gen_matrix.cuda()

        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros((self.del_num * (2 * self.sz - self.del_num - 1) // 2, 2))
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh

        out = out[:, 0]

        if use_cuda:
            out = out.cuda()

        matrix = torch.zeros(self.sz, self.sz).cuda()
        left_mask = torch.ones(self.sz, self.sz)
        left_mask[:-self.del_num, :-self.del_num] = 0
        left_mask = left_mask - torch.diag(torch.diag(left_mask))
        un_index = torch.triu(left_mask).nonzero()

        matrix[(un_index[:, 0], un_index[:, 1])] = out
        out_matrix = matrix + matrix.T
        # out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)


'''Generate unknown part of continuous node state'''
class Generator_states(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)

    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        return pos_probs

'''Generate unknown part of discrete node state'''
class Generator_states_discrete(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states_discrete, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)

    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        probs = torch.cat([pos_probs, 1 - pos_probs], 2)
        return probs


#############
# Functions #
#############
def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    if use_cuda:
        gumbel = gumbel.cuda()
    return gumbel


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y