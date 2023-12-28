import time
import torch.nn.utils as U
import torch.optim as optim
from model_2 import *
from tools import *
import argparse
import logging


# configuration
HYP = {
    'node_size': 6,
    'hid': 128,  # hidden size
    'epoch_num': 1000,  # epoch 1000
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
parser.add_argument('--nodes', type=int, default=6, help='Number of nodes, default=10')
parser.add_argument('--network', type=str, default='ER', help='type of network')
parser.add_argument('--sys', type=str, default='spring', help='simulated system to model,spring or cmn')
parser.add_argument('--dim', type=int, default=1, help='# information dimension of each node spring:4 cmn:1 ')
parser.add_argument('--exp_id', type=int, default=1, help='experiment_id, default=1')
parser.add_argument('--device_id', type=int, default=-1, help='Gpu_id, default=5')
args = parser.parse_args()
#set gpu id
torch.cuda.set_device(args.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f'start_time: {start_time}')

# model load  path
gen_path = f"./model_lab/gen_{args.network}_{args.nodes}.pkl"
dyn_path = f"./model_lab/dyn_{args.network}_{args.nodes}.pkl"
generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

# load_data
def load_lab(batch_size=128):
    data_path = '/Users/gxz/Desktop/PT/复杂系统/data/demo/lab_checks_array.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    sample_cnt = data.shape[0]
    data = torch.from_numpy(data.astype(np.float32))
    data = data.transpose(1, 2)

    train_cnt = int(sample_cnt*8/10)
    # train = data[:train_cnt]
    test = data[train_cnt:]
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return test_loader

test_loader = load_lab(batch_size=HYP['batch_size'])


def test_dyn_gen():
    loss_batch = []
    mse_batch = []

    for idx, data in enumerate(test_loader):
        logging.info(f"batch idx: {idx}")
        # data
        data = data.to(device)
        x = data[:, :, 0, :]
        y = data[:, :, 1, :]
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1), 1)
        temp_x = x

        for batch_i in range(x.size(0)):
            predict_step = int(y[batch_i, 0, 0])
            for s in range(predict_step):
                cur_temp_x = temp_x[[batch_i], :, 1:]
                adj = generator.sample_all().to(device)
                y_hat = dyn_isom(cur_temp_x, adj)
                temp_x[batch_i, :, 1:] = y_hat
            outputs[[batch_i], :, :] = temp_x[[batch_i], :, 1:]

        loss = torch.mean(torch.abs(outputs - y[:, :, 1:].cpu()))
        loss_batch.append(loss.item())
        mse_batch.append(F.mse_loss(y[:, :, 1:].cpu(), outputs).item())

    return np.mean(loss_batch), np.mean(mse_batch),


if __name__ == '__main__':
    with torch.no_grad():
        loss, mse = test_dyn_gen()
        logging.info(f"finish loss: {str(loss)} --> mse: {str(mse)}")