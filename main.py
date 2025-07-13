import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from utils.utils import load_config
import random
import numpy as np
import torch
from utils import save
from utils.data_model import read_data
from models.get_models import get_model 
from utils.loss import LF
from utils.trainer import Trainer
import argparse
import torch.backends.cudnn as cudnn
from utils.data_ingestion_and_merge import ingestion

def seed_everything(seed=2020):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='./config/ContextNet_outbrain', help='The config directory.')
    parser.add_argument('--model_setid', type=str, default='base', help='The model experiment id to run.')
    parser.add_argument('--data_setid', type=str, default='base', help='The data experiment id to run.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--disable_cuda', default=False, help='Disable CUDA')
    parser.add_argument('--gpu_index', default=1, type=int, help='Gpu index.')
    parser.add_argument('--expid', type=str, default='v1', help='Write training log to file.')

    args = parser.parse_args()
    model_params, data_params = load_config(args)
    args.model_params = model_params
    args.data_params = data_params

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda:' + str(args.gpu_index))
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    log_path = './result/' + str(model_params['model_name']) + '/' + str(data_params['data_name']) + '/' + str(args.expid) + '/'
    args.log_path = log_path
    save.ensure_dir(args.log_path, verbose=True)
    save.save_config(args, args.log_path + 'config.json', verbose=True)
    file_logger = save.FileLogger(log_path + 'log.txt',
                                  header="epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
    args.file_logger = file_logger
    seed_everything(args.seed)

    ingestion()

    train_loader, test_loader, valid_loader, fix_SparseFeat, fix_DenseFeat = read_data(args)
    print('finish data read')
    args.file_logger.log('Data loading complete')

    model = get_model(fix_SparseFeat, fix_DenseFeat, args)
    print(f"Using model: {type(model).__name__}")      
    loss_function = LF(model, args)

    with torch.cuda.device(args.gpu_index):
        for batch_x, batch_y in train_loader:
            print("Sample batch input shape:", batch_x.shape)  
            break
        trainer = Trainer(model, args, loss_function, train_loader, test_loader, valid_loader)
        trainer.train()
