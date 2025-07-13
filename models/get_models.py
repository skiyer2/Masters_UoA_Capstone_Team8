from .ContextNet import SessionContextNet
import torch


def get_model(fix_SparseFeat, fix_DenseFeat,args):
    if args.model_params['model_name'] == 'SessionContextNet':
         model = SessionContextNet(fix_SparseFeat, fix_DenseFeat, args).to(args.device)
    else:
        raise ValueError('Unexpected model name')
    return model