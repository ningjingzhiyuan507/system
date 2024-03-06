import time
from tools import *
import argparse
import logging


# configuration
HYP = {
    'epoch_num': 1000,  # epoch 1000
    'batch_size': 16,  # batch size 512
    'temp': 1,  # temperature
}


parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=210, help='Number of nodes, default=10')
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

# 并行初始化
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.info(f'start_time: {start_time}')

# model load  path
gen_path = f"/code/AIDD/AIDD/model_lab/gen_ER_{args.nodes}.pkl"
dyn_path = f"/code/AIDD/AIDD/model_lab/dyn_ER_{args.nodes}.pkl"
generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

# load_data
def load_lab(batch_size=128):
    data_path = '/code/AIDD/AIDD/lab_test.pickle'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    data = torch.from_numpy(data.astype(np.float32))
    data = data.transpose(1, 2)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

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
            outputs[[batch_i], :, :] = temp_x[[batch_i], :, 1:].cpu()

        loss = torch.mean(torch.abs(outputs - y[:, :, 1:].cpu()))
        loss_batch.append(loss.item())
        mse_batch.append(F.mse_loss(y[:, :, 1:].cpu(), outputs).item())

    return np.mean(loss_batch), np.mean(mse_batch),


if __name__ == '__main__':
    with torch.no_grad():
        loss, mse = test_dyn_gen()
        logging.info(f"finish loss: {str(loss)} --> mse: {str(mse)}")