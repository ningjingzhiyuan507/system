import sys
sys.path.append('/Users/gxz/pt/AIDD')

import time
import torch.nn.utils as U
import torch.optim as optim
from model_2 import *
from tools import *
import argparse
import logging


# configuration
HYP = {
    'hid': 128,  # hidden size
    'epoch_num': 1000,  # epoch 1000
    'batch_size': 128,  # batch size 512
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

# generator（网络生成器）
generator = Gumbel_Generator_Old(
    sz=args.nodes, temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
# generator optimizer
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# dyn learner（动态学习器）
dyn_isom = IO_B(args.dim, HYP['hid']).to(device)
# dyn learner optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])

# load_data
def load_lab(batch_size=128):
    data_path = '/Users/gxz/Desktop/PT/复杂系统/data/第三版/lab_train.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data = torch.from_numpy(data.astype(np.float32))
    data = data.transpose(1, 2)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_loader

train_loader = load_lab(batch_size=HYP['batch_size'])


def train_dyn_gen():
    loss_batch = []
    mse_batch = []
    for idx, data in enumerate(train_loader):
        logging.info(f"batch idx: {idx}")  # 每次抽取一组batch_size大小的数据进行训练
        # data
        data = data.to(device)  # samples, nodes, P, dim
        x = data[:, :, 0, :]  # 将第一个时间节点作为x [samples, nodes, dim]
        y = data[:, :, 1, :]  # 将第二个时间节点作为y [samples, nodes, dim]
        # drop temperature
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 1)  # samples, nodes, dim(value)
        temp_x = x

        op_net.zero_grad()
        op_dyn.zero_grad()

        for batch_i in range(x.size(0)):  # 每个sample时间间隔不等，需要逐sample进行predict
            predict_step = int(y[batch_i, 0, 0])
            for s in range(predict_step):  # 同一个sample所有nodes时间间隔相同
                cur_temp_x = temp_x[[batch_i], :, 1:]  # 必须是3维 sample, nodes, dim
                adj = generator.sample_all().to(device)
                y_hat = dyn_isom(cur_temp_x, adj)
                temp_x[batch_i, :, 1:] = y_hat

            outputs[[batch_i], :, :] = temp_x[[batch_i], :, 1:]

        loss = torch.mean(torch.abs(outputs - y[:, :, 1:].cpu()))
        loss.backward()  # backward and optimize
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
    # start training
    best_val_mse = 1000000
    best = 0
    best_loss = 10000000

    # model save path
    dyn_path = f'./model_lab/dyn_{args.network}_{str(args.nodes)}.pkl'
    gen_path = f'./model_lab/gen_{args.network}_{str(args.nodes)}.pkl'
    adj_path = f'./model_lab/adj_{args.network}_{str(args.nodes)}.pkl'

    # each training epoch
    for e in range(HYP['epoch_num']):
        logging.info(f"\nepoch: {e}")
        t_s = time.time()
        try:
            # train both dyn learner and generator together
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
            torch.save(dyn_isom, dyn_path)  # 保存最优动力学模型(IO_B(n2e,e2e,n2n,output))
            torch.save(generator, gen_path)  # 保存最优网络生成器(结果Gumbel_Gnerator.Old())
            out_matrix = generator.sample_all(hard=HYP['hard_sample'], ).to(device)
            torch.save(out_matrix, adj_path)  # 从网络生成器中提取邻接矩阵并保存。预测时不需要该矩阵（里面的值为预测为1时的概率）
        logging.info(f"best epoch: {best}")
        t_e = time.time()
        logging.info(f"time for this whole epoch: {str(round(t_e - t_s, 2))}")

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"end time: {end_time}")
