"""单机多卡并行训练"""
import sys
sys.path.append('/code/AIDD')

import time
import torch.nn.utils as U
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler

from model_2 import *
from tools import *
import argparse
import logging


# configuration
HYP = {
    'hid': 128,  # hidden size
    'epoch_num': 1000,  # epoch 1000
    'batch_size': 8,  # batch size 512
    'lr_net': 0.004,  # lr for net generator 0.004
    'lr_dyn': 0.001,  # lr for dyn learner
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=210, help='Number of nodes, default=10')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node spring:4 ')
args = parser.parse_args()

# 并行初始化
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f"start_time: {start_time}")

# 网络生成器
generator = Gumbel_Generator_Old(
    sz=args.nodes, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator = torch.nn.parallel.DistributedDataParallel(
    generator,
    device_ids=[local_rank],
    output_device=local_rank
)
generator.init(0, 0.1)
# generator optimizer
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# 动态学习器
dyn_isom = IO_B(args.dim, HYP['hid']).to(device)
dyn_isom = torch.nn.parallel.DistributedDataParallel(
    dyn_isom,
    device_ids=[local_rank],
    output_device=local_rank
)
# dyn learner optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])


def load_lab(batch_size=128):
    data_path = '/code/AIDD/AIDD/lab_test.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data = torch.from_numpy(data.astype(np.float32))
    data = data.transpose(1, 2)
    train_loader = DataLoader(  # 并行化抽取，各进程分别喂数据
        data, batch_size=batch_size, sampler=DistributedSampler(data))

    return train_loader


def train_dyn_gen(data_loader):
    loss_batch = []
    mse_batch = []
    for idx, data in enumerate(data_loader):
        logging.info(f"batch idx: {idx}")
        x = data[:, :, 0, :]
        y = data[:, :, 1, :]

        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 1)
        temp_x = x

        op_net.zero_grad()
        op_dyn.zero_grad()

        for batch_i in range(x.size(0)):
            predict_step = int(y[batch_i, 0, 0])
            for s in range(predict_step):
                cur_temp_x = temp_x[[batch_i], :, 1:]
                adj = generator.sample_all().to(device)
                y_hat = dyn_isom(cur_temp_x, adj)
                temp_x[batch_i, :, 1:] = y_hat

            outputs[[batch_i], :, :] = temp_x[[batch_i], :, 1:].cpu()

        loss = torch.mean(torch.abs(outputs - y[:, :, 1:].cpu()))
        loss.backward()
        U.clip_grad_norm_(generator.gen_matrix, 0.000075)

        op_net.step()
        op_dyn.step()

        loss_batch.append(loss.item())
        mse_batch.append(F.mse_loss(y[:, :, 1:].cpu(), outputs).item())

    op_net.zero_grad()
    loss = (torch.sum(generator.sample_all())) * HYP['lr_stru']
    loss.backward()
    op_net.step()

    return np.mean(loss_batch), np.mean(mse_batch)


if __name__ == '__main__':
    best_val_mse = 1000000
    best = 0
    best_loss = 10000000

    dyn_path = f'/code/AIDD/AIDD/model_lab/dyn_ER_{str(args.nodes)}.pkl'
    gen_path = f'/code/AIDD/AIDD/model_lab/gen_ER_{str(args.nodes)}.pkl'
    adj_path = f'/code/AIDD/AIDD/model_lab/adj_ER_{str(args.nodes)}.pkl'

    train_loader = load_lab(batch_size=HYP['batch_size'])
    for e in range(HYP['epoch_num']):
        logging.info(f"\nepoch: {e}")
        t_s = time.time()

        train_loader.sampler.set_epoch(e)  # 设定seed保持各进程同步
        try:
            loss, mse = train_dyn_gen()
        except RuntimeError as sss:
            if 'out of memory' in str(sss):
                logging.warning('|WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise sss

        logging.info(f"loss: {str(loss)} mse: {str(mse)}")

        if loss < best_loss:
            logging.info(f"best epoch: {e}")
            best_loss = loss
            best = e
            if torch.distributed.get_rank() == 0:  # 保存一个即可
                torch.save(dyn_isom, dyn_path)
                torch.save(generator, gen_path)
                out_matrix = generator.sample_all(hard=HYP['hard_sample'], ).to(device)
                torch.save(out_matrix, adj_path)
        logging.info(f"best epoch: {best}")
        t_e = time.time()
        logging.info(f"time for this whole epoch: {str(round(t_e - t_s, 2))}")

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"end time: {end_time}")
