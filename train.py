import argparse
import datasets
from model import trainers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1.0.0-stage1')
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--n_cls', type=int, default=57)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--npx', type=int, default=64, help='Output pixel size')
    parser.add_argument('--nvx', type=int, default=32, help='Output voxel size')
    parser.add_argument('--dataset_path', type=str, default='./dataset/shapenet/*/*/*.binvox')
    parser.add_argument('--stage1_params_path', type=str, default='', help='You need this arg when you train stage2')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Stage 1
    dataset = datasets.Stage1(args)
    trainer = trainers.Stage1(args)
    trainer.run(args, dataset)
    trainer.close()

    # Stage 2
    # dataset = datasets.Stage2(args)
    # trainer = trainers.Stage2(args)
    # trainer.run(args, dataset)
    # trainer.close()
