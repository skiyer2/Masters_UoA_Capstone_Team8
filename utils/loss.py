import torch
import torch.nn.functional as F
from torch import nn, optim

class LF:
    def __init__(self, recmodel, args):
        self.model = recmodel
        self.lr = args.model_params['learning_rate']
        if args.model_params['optimizer'] == 'adam':
            self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        elif args.model_params['optimizer'] == 'SGD':
            self.opt = optim.SGD(recmodel.parameters(), lr=self.lr)
        elif args.model_params['optimizer'] == "adagrad":
            self.opt = optim.Adagrad(recmodel.parameters(), lr=self.lr)
        elif args.model_params['optimizer'] == "rmsprop":
            self.opt = optim.RMSprop(recmodel.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        if args.model_params['loss'] == "binary_crossentropy":
            self.loss_func = F.binary_cross_entropy
        elif args.model_params['loss'] == "mse":
            self.loss_func = nn.MSELoss()
        elif args.model_params['loss'] == "mae":
            self.loss_func = nn.L1Loss()
        else:
            raise NotImplementedError

    def stageOne(self, x, y):
        y_pred, reg_loss = self.model(x)
        y_pred_flat = y_pred.view(-1)
        y_flat = y.view(-1)
        mask = (y_flat != -1)
        if mask.sum() == 0:
            loss = reg_loss
        else:
            loss = self.loss_func(y_pred_flat[mask], y_flat[mask]) + reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item()
