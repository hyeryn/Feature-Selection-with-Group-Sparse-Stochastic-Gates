import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import argparse

from exp.exp_main_itr import Exp_Main

def main():

    fix_seed = 2024

    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(fix_seed)
    
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--id', default=0, help='number of experiment test')
    parser.add_argument('--data', type=str, default='syn', help='dataset, options: [syn, gas, breast(2), pbmc]')
    parser.add_argument('--group_type', type=str, default='chrom', 
                        help='for only pbmc dataset : group select [chrom,loc]')
    parser.add_argument('--dtype', type=str, default='', help='synthetic dataset, options: [A,B,C,D,E ...]')
    parser.add_argument('--selector', type=str, default='HC', help='selector, options: [HC,STG]')
    parser.add_argument('--distribution', type=str, default='norm', 
                        help='poisson distribution approximation [poi:poisson, norm:normal, rnom:refined-normal]')
    parser.add_argument('--checkpoints', type=str, 
                        default='/mnt/storage/personal/hrpark/personal/groupsparsity/checkpoints/', help='location of model checkpoints')

    # regularization for feature selection
    parser.add_argument('--lamb1', default=0, type=float, help='L0 penalty(individual sparsity)')
    parser.add_argument('--lamb2', default=0, type=float, help='L2 penalty(group sparsity)')
    parser.add_argument('--threshold', default=0.3, type=float, 
                        help='consider probability below a certain value to be zero, option: [0.3, 0.5, 0.7]')

    # optimization
    parser.add_argument('--epochs', default=200, type=int, help='train epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size of train input data')
    parser.add_argument('--lr', default=1e-5, type=float, help='model learning rate')
    parser.add_argument('--lr2', default=1e-2, type=float, help='feature selector learning rate')
    parser.add_argument('--patience', type=int, default=20, 
                        help='early stopping patience [default:no early stop = same with epochs]')    
    parser.add_argument('--lradj', type=str, default='type0', 
                        help='adjust learning rate [type0:no, type1:exp down, type2:grid search]')
    parser.add_argument('--w_dec1', default=0.01, type=float, help='network first layer w norm (for mask dominant)')
    parser.add_argument('--w_dec2', default=0.001, type=float, help='network except first layer w norm')
    parser.add_argument('--permuted', default='permuted', type=str, help='[None, permuted]')
    parser.add_argument('--norm', default=0, type=float, help='group sparsity normalize term')
    
    # gpu
    parser.add_argument('--device', default='cuda:6')

    args = parser.parse_args()
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    exp = Exp(args)

    if args.norm == 0:
        if args.permuted == None:
            if args.selector == 'HC':
                setting = '{}_data{}{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}'.format(
                    args.id,
                    args.data,
                    args.dtype,
                    args.selector,
                    args.distribution,
                    args.lamb1,
                    args.lamb2,
                    args.lradj,
                    args.w_dec1,
                    args.w_dec2
                )
            else:
                setting = '{}_data{}{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}'.format(
                    args.id,
                    args.data,
                    args.dtype,
                    args.selector,
                    args.lamb1,
                    args.lamb2,
                    args.lradj,
                    args.w_dec1,
                    args.w_dec2
                )
        else:
            if args.data == 'pbmc':
                if args.selector == 'HC':
                    setting = '{}_data{}{}_{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}_{}'.format(
                        args.id,
                        args.data,
                        args.group_type,
                        args.dtype,
                        args.selector,
                        args.distribution,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted
                    )
                else:
                    setting = '{}_data{}{}_{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}_{}'.format(
                        args.id,
                        args.data,
                        args.group_type,
                        args.dtype,
                        args.selector,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted
                    )
            else:
                if args.selector == 'HC':
                    setting = '{}_data{}{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}_{}'.format(
                        args.id,
                        args.data,
                        args.dtype,
                        args.selector,
                        args.distribution,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted
                    )
                else:
                    setting = '{}_data{}{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}_{}'.format(
                        args.id,
                        args.data,
                        args.dtype,
                        args.selector,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted
                    )
    else:
        if args.permuted == None:
            if args.selector == 'HC':
                setting = '{}_data{}{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}_n{}'.format(
                    args.id,
                    args.data,
                    args.dtype,
                    args.selector,
                    args.distribution,
                    args.lamb1,
                    args.lamb2,
                    args.lradj,
                    args.w_dec1,
                    args.w_dec2,
                    args.norm
                )
            else:
                setting = '{}_data{}{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}_n{}'.format(
                    args.id,
                    args.data,
                    args.dtype,
                    args.selector,
                    args.lamb1,
                    args.lamb2,
                    args.lradj,
                    args.w_dec1,
                    args.w_dec2,
                    args.norm
                )
        else:
            if args.data == 'pbmc':
                if args.selector == 'HC':
                    setting = '{}_data{}{}_{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}_{}_n{}'.format(
                        args.id,
                        args.data,
                        args.group_type,
                        args.dtype,
                        args.selector,
                        args.distribution,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted,
                        args.norm
                    )
                else:
                    setting = '{}_data{}{}_{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}_{}_n{}'.format(
                        args.id,
                        args.data,
                        args.group_type,
                        args.dtype,
                        args.selector,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted,
                        args.norm
                    )
            else:
                if args.selector == 'HC':
                    setting = '{}_data{}{}_sel{}{}_l1_{}_l2_{}_{}_wdec{}_{}_{}_n{}'.format(
                        args.id,
                        args.data,
                        args.dtype,
                        args.selector,
                        args.distribution,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted,
                        args.norm
                    )
                else:
                    setting = '{}_data{}{}_sel{}_l1_{}_l2_{}_{}_wdec{}_{}_{}_n{}'.format(
                        args.id,
                        args.data,
                        args.dtype,
                        args.selector,
                        args.lamb1,
                        args.lamb2,
                        args.lradj,
                        args.w_dec1,
                        args.w_dec2,
                        args.permuted,
                        args.norm
                    )

    print('>>>> start training : {} <<<<'.format(setting))
    exp.train(setting)
    #print('>>>> testing <<<<')
    #exp.test(setting)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()