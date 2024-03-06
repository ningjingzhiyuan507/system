import sys
sys.path.append('/code/AIDD')

from tools import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()

# 并行初始化
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)

adj_path = "/code/AIDD/AIDD/model_lab/gen_ER_210.pkl"
gen_path = "/code/AIDD/AIDD/model_lab/gen_ER_210.pkl"
dyn_path = "/code/AIDD/AIDD/model_lab/dyn_ER_210.pkl"

adj = torch.load(adj_path).to(device)
generator = torch.load(gen_path).to(device)
dyn_isom = torch.load(dyn_path).to(device)

torch.save(adj.module, '/code/AIDD/AIDD/model_lab/gen_ER_210_module.pkl')
torch.save(generator.module, '/code/AIDD/AIDD/model_lab/gen_ER_210_module.pkl')
torch.save(dyn_isom.module, '/code/AIDD/AIDD/model_lab/gen_ER_210_module.pkl')