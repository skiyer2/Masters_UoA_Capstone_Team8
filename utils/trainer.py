import logging
import os
from scipy import sparse
import sys
import torch
import torch.nn.functional as F
from utils.save import save_checkpoint
from utils.evaluate import get_auc
from sklearn.metrics import log_loss
import numpy as np
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm
                    
class Trainer():
    def __init__(self, model, args, loss_function, train_loader, test_loader, valid_loader):
        self.model = model
        self.loss_function = loss_function
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        self.best_results = {'auc': np.zeros(1), 'logloss': float('inf')}

    def train(self):
        print('Train Begin')
        Recmodel = self.model
        for epoch_counter in range(self.args.model_params['epochs']):
            Recmodel.train()
            aver_loss = 0.
            for batch_counter in tqdm(self.train_loader):
                x = batch_counter[0].long().to(self.args.device)
                y = batch_counter[1].float().to(self.args.device)
                cri = self.loss_function.stageOne(x, y)
                aver_loss += cri

            aver_loss = aver_loss / len(self.train_loader)
            print('Epoch:', epoch_counter, '|loss:%.4f' % aver_loss)
            self.args.file_logger.log("{}\t{:.6f}".format(epoch_counter, aver_loss))
            if self.args.model_params['eval_epoch'] is not None:
                if epoch_counter % self.args.model_params['eval_epoch'] == 0:
                    self.valid(epoch_counter)
        self.test(epoch_counter)

    def valid(self, epoch_counter):
        self.model.eval()
        with torch.no_grad():
            auc = 0
            total_logloss = 0
            count = 0
            for batch_counter in tqdm(self.valid_loader):
                x = batch_counter[0].long().to(self.args.device)
                y = batch_counter[1].float().to(self.args.device)
                y_pred, _ = self.model(x)
                y_pred_flat = y_pred.view(-1).cpu().detach().numpy()
                y_flat = y.view(-1).cpu().detach().numpy()
                mask = (y_flat != -1)
                if mask.sum() == 0:
                    continue
                auc += get_auc(y_pred_flat[mask], y_flat[mask])
                try:
                    total_logloss += log_loss(y_flat[mask], y_pred_flat[mask], labels=[0, 1])
                except Exception as e:
                    print("Log loss error:", e)
                count += 1

            avg_auc = auc / max(1, count)
            avg_logloss = total_logloss / max(1, count)
            print(f"Validation AUC: {avg_auc:.6f}, LogLoss: {avg_logloss:.6f}")
            self.args.file_logger.log(f"Validation AUC: {avg_auc:.6f}, LogLoss: {avg_logloss:.6f}")

            if avg_auc > self.best_results['auc'] or avg_logloss < self.best_results['logloss']:
                self.best_results['auc'] = avg_auc
                self.best_results['logloss'] = avg_logloss
                self.best_results['best_epoch'] = epoch_counter
                self.test(epoch_counter)
            print(self.best_results)
            self.args.file_logger.log(str(self.best_results))

    def test(self, epoch_counter):
        self.model.eval()
        with torch.no_grad():
            auc = 0
            total_logloss = 0
            count = 0
            for batch_counter in tqdm(self.test_loader):
                x = batch_counter[0].long().to(self.args.device)
                y = batch_counter[1].float().to(self.args.device)
                y_pred, _ = self.model(x)
                y_pred_flat = y_pred.view(-1).cpu().detach().numpy()
                y_flat = y.view(-1).cpu().detach().numpy()
                mask = (y_flat != -1)
                if mask.sum() == 0:
                    continue
                auc += get_auc(y_pred_flat[mask], y_flat[mask])
                try:
                    total_logloss += log_loss(y_flat[mask], y_pred_flat[mask], labels=[0, 1])
                except Exception as e:
                    print("Log loss error:", e)
                count += 1

            avg_auc = auc / max(1, count)
            avg_logloss = total_logloss / max(1, count)
            print(f"Test AUC: {avg_auc:.6f}, LogLoss: {avg_logloss:.6f}")
            self.args.file_logger.log(f"Test AUC: {avg_auc:.6f}, LogLoss: {avg_logloss:.6f}")
