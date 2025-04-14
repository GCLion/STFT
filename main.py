import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver
import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    np.random.seed(42)
    random_float = np.random.random()
    print(random_float)
    parser = argparse.ArgumentParser()
# 1 ADM 
    parser.add_argument('--lr', type=float, default=1e-4) 
    parser.add_argument('--num_epochs', type=int, default=120) 
    parser.add_argument('--k', type=int, default=1) 
    parser.add_argument('--data_path', type=str, default='./dataset/stable_diffusion_v_1_4_train.csv') 
    parser.add_argument('--data_path_', type=str, default='./dataset/stable_diffusion_v_1_4_test.csv') 
    parser.add_argument('--win_size', type=int, default=19)
    parser.add_argument('--input_c', type=int, default=20)#192 3*256=768
    parser.add_argument('--output_c', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)#256
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=50.0)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
