import torch
import os, argparse
#from srcnn import SRCNN
from vdsr import VDSR
from fsrcnn import FSRCNN
#from srgan import SRGAN
#from drcn import DRCN
#from espcn import ESPCN
#from edsr import EDSR
#from lapsrn import LapSRN

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of SR collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='VDSR',
                        choices=['SRCNN', 'VDSR', 'DRCN', 'ESPCN', 'FastNeuralStyle', 'FSRCNN', 'SRGAN', 'LapSRN',
                                 'EnhanceNet', 'EDSR', 'EnhanceGAN'], help='The type of model')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/Downloads/challenge-sat/probav_data')
    parser.add_argument('--train_dataset', type=list, default=['DIV2K'], choices=['bsds300', 'General100', 'T91'],
                        help='The name of training dataset')
    parser.add_argument('--test_dataset', type=list, default=['Set5', 'Set14', 'Urban100'], choices=['Set5', 'Set14', 'Urban100'],
                        help='The name of test dataset')
    parser.add_argument('--crop_size', type=int, default=384, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--scale_factor', type=int, default=2, help='Size of scale factor')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--save_dir', type=str, default='Result_training', help='Directory name to save the results')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    if args.model_name == 'SRCNN':
        net = SRCNN(args)
    elif args.model_name == 'VDSR':
        net = VDSR(args)
    elif args.model_name == 'DRCN':
        net = DRCN(args)
    elif args.model_name == 'ESPCN':
        net = ESPCN(args)
    # elif args.model_name == 'FastNeuralStyle':
    #     net = FastNeuralStyle(args)
    elif args.model_name == 'FSRCNN':
        net = FSRCNN(args)
    elif args.model_name == 'SRGAN':
        net = SRGAN(args)
    elif args.model_name == 'LapSRN':
        net = LapSRN(args)
    # elif args.model_name == 'EnhanceNet':
    #     net = EnhanceNet(args)
    elif args.model_name == 'EDSR':
        net = EDSR(args)
    # elif args.model_name == 'EnhanceGAN':
    #     net = EnhanceGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)

    # train
    net.train()

    # test
    net.test()
    # net.test_single('getchu_full.jpg')

if __name__ == '__main__':
    main()