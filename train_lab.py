import sys
sys.path.append('/Users/gxz/Desktop/PT/复杂系统/AIDD')

import numpy as np
import pickle

import torch
import time
import torch.nn.utils as U
import torch.optim as optim
from model import *
from tools import *
import argparse
import logging

# configuration
HYP = {
    'node_size': 124,
    'hid': 128,  # hidden size
    'epoch_num': 1,  # epoch 1000
    'batch_size': 512,  # batch size 512
    'lr_net': 0.004,  # lr for net generator 0.004
    'lr_dyn': 0.001,  # lr for dyn learner
    'lr_stru': 0.0001,  # lr for structural loss 0.0001 2000 0.01  0.00001
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'temp': 1,  # temperature
    'drop_frac': 1,  # temperature drop frac
}


parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=124, help='Number of nodes, default=10')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node spring:4 ')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=-1, help='Gpu_id, default=5')  # default默认5
args = parser.parse_args()

#set gpu id
torch.cuda.set_device(args.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f"start_time: {start_time}")

generator = Gumbel_Generator_Old(
    sz=args.nodes, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

dyn_isom = IO_B(args.dim, HYP['hid']).to(device)
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])


def load_lab(batch_size=128):
    data_path = 'AIDD/AIDD/lab_array.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    sample_cnt = data.shape[0]
    data = torch.from_numpy(data.astype(np.float32))
    data = data.transpose(1, 2)

    train_cnt = int(sample_cnt*8/10)
    train = data[:train_cnt]
    test = data[train_cnt:]
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=20)

    return train_loader, test_loader

train_loader, _ = load_lab(batch_size=HYP['batch_size'])


def train_dyn_gen():
    loss_batch = []
    mse_batch = []
    logging.info(f"current temp: {generator.temperature}")
    for idx, data in enumerate(train_loader):
        logging.info(f"batch idx: {idx}")

        data = data.to(device)
        x = data[:, :, 0, :]
        y = data[:, :, 1, :]
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 1)
        temp_x = x

        op_net.zero_grad()
        op_dyn.zero_grad()

        num = int(args.nodes / HYP['node_size'])
        remainder = int(args.nodes % HYP['node_size'])
        if remainder == 0:
            num = num - 1

        for batch_i in range(x.size(0)):
            predict_step = int(y[batch_i, 0, 0])
            for s in range(predict_step):
                cur_temp_x = temp_x[[batch_i], :, 1:]
                for j in range(args.nodes):
                    adj_col = generator.sample_adj_i(
                        j, hard=HYP['hard_sample'],
                        sample_time=HYP['sample_time']).to(device)
                    y_hat = dyn_isom(
                        cur_temp_x, adj_col, j, num, HYP['node_size'])
                    temp_x[batch_i, j, 1:] = y_hat
            outputs[[batch_i], :, :] = temp_x[[batch_i], :, 1:]

        loss = torch.mean(torch.abs(outputs - y[:, :, 1:].cpu()))
        loss.backward()

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

    dyn_path = f'./model_lab/dyn_{args.network}_{str(args.nodes)}.pkl'
    gen_path = f'./model_lab/gen_{args.network}_{str(args.nodes)}.pkl'
    adj_path = f'./model_lab/adj_{args.network}_{str(args.nodes)}.pkl'

    for e in range(HYP['epoch_num']):
        logging.info(f"\nepoch: {e}")
        t_s = time.time()
        t_s1 = time.time()
        try:
            loss, mse = train_dyn_gen()
        except RuntimeError as sss:
            if 'out of memory' in str(sss):
                logging.info('|WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise sss

        t_e1 = time.time()
        logging.info(f"loss: {str(loss)} mse: {str(mse)}")
        logging.info(f"time for this dyn_adj epoch: {str(round(t_e1 - t_s1, 2))}")

        if loss < best_loss:
            logging.info(f"best epoch: {e}")
            best_loss = loss
            best = e
            torch.save(dyn_isom,  dyn_path)
            torch.save(generator, gen_path)
            out_matrix = generator.sample_all(hard=HYP['hard_sample'], ).to(
                device)
            torch.save(out_matrix, adj_path)
        logging.info(f"best epoch: {best}")
        t_e = time.time()
        logging.info(f"time for this whole epoch: {str(round(t_e - t_s, 2))}")

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"end time: {end_time}")
