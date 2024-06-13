import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import Optimizer
import numpy as np

def train(model:nn.Module, data_loader: DataLoader, optimizer: Optimizer,
                criterion, device, batch_size):
    # 0-2: pred, 3: utt_id
    infos = np.empty([len(data_loader) * batch_size, 4])
    batch_loss = []
    model.train()
    for step, batch in enumerate(data_loader):
        # for readibility 
        batch_X = batch[0]
        batch_Y = batch[1]
        seq_len = batch[2]
        utt_ids = batch[3]

        # sort for gru
        sorted_index = torch.argsort(-seq_len)
        batch_X = batch_X[sorted_index]
        batch_Y = batch_Y[sorted_index]
        seq_len = seq_len[sorted_index]
        utt_ids = utt_ids[sorted_index]
        
        # step and otimize model
        model.zero_grad(set_to_none=True)
        _, pred = model.forward(batch_X.to(device), seq_len.to(device))
        loss = criterion(pred, batch_Y.to(device))
        loss_weighted = torch.mean(loss)
        loss_weighted.backward()
        optimizer.step()
        
        infos[step * batch_size : (step+1) * batch_size, 0:3] = \
            pred.detach().clone().cpu().numpy()
        infos[step * batch_size : (step+1) * batch_size, 3] = \
            utt_ids.detach().clone().numpy()
        batch_loss.append(loss_weighted.detach().clone().cpu().tolist())
        torch.cuda.empty_cache()
    return infos, batch_loss

def run(model: nn.Module, data_loader: DataLoader, device, batch_size, data_size):
        # 0-2: pred, 3: utt_id
    infos = np.empty([data_size, 4])
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # for readibility
            batch_X = batch[0]
            batch_Y = batch[1]
            seq_len = batch[2]
            utt_ids = batch[3]
        
            # sort for gru
            sorted_index = torch.argsort(-seq_len)
            batch_X = batch_X[sorted_index]
            batch_Y = batch_Y[sorted_index]
            seq_len = seq_len[sorted_index]
            utt_ids = utt_ids[sorted_index]

            # step and otimize model
            _, pred = model.forward(batch_X.to(device), seq_len.to(device))
        
            infos[step * batch_size : (step+1) * batch_size, 0:3] = \
                pred.detach().clone().cpu().numpy()
            infos[step * batch_size : (step+1) * batch_size, 3] = \
                utt_ids.detach().clone().numpy()
            torch.cuda.empty_cache()
    return infos

def fair_train(model:nn.Module, data_loader: DataLoader, optimizer: Optimizer,
                criterion, fairness, device, batch_size, lam):
    # 0-2: pred, 3: utt_id
    infos = np.empty([len(data_loader) * batch_size, 4])
    batch_loss = []
    model.train()
    for step, batch in enumerate(data_loader):
        # for readibility 
        batch_X = batch[0]
        batch_Y = batch[1]
        seq_len = batch[2]
        utt_ids = batch[3]
        protected = batch[4]

        # sort for gru
        sorted_index = torch.argsort(-seq_len)
        batch_X = batch_X[sorted_index]
        batch_Y = batch_Y[sorted_index]
        seq_len = seq_len[sorted_index]
        utt_ids = utt_ids[sorted_index]
        protected = protected[sorted_index]
        
        # step and otimize model
        model.zero_grad(set_to_none=True)
        _, pred = model.forward(batch_X.to(device), seq_len.to(device))
        loss = criterion(pred, batch_Y.to(device))
        fair = fairness(pred, protected.float().to(device))
        loss = loss + lam * fair
        loss_weighted = torch.mean(loss)
        loss_weighted.backward()
        optimizer.step()
        
        infos[step * batch_size : (step+1) * batch_size, 0:3] = \
            pred.detach().clone().cpu().numpy()
        infos[step * batch_size : (step+1) * batch_size, 3] = \
            utt_ids.detach().clone().numpy()

        batch_loss.append(loss_weighted.detach().clone().cpu().tolist())
        torch.cuda.empty_cache()
    return infos, batch_loss

def emb_run(model: nn.Module, data_loader: DataLoader, device, batch_size, data_size):
        # 0-2: pred, 3: utt_id
    infos = np.empty([data_size, 4])
    hidden_embs = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            # for readibility
            batch_X = batch[0]
            batch_Y = batch[1]
            seq_len = batch[2]
            utt_ids = batch[3]
        
            # sort for gru
            sorted_index = torch.argsort(-seq_len)
            batch_X = batch_X[sorted_index]
            batch_Y = batch_Y[sorted_index]
            seq_len = seq_len[sorted_index]
            utt_ids = utt_ids[sorted_index]

            # step and otimize model
            tfm_embs, pred = model.forward(batch_X.to(device), seq_len.to(device))

            embs = []
            for i, data in enumerate(tfm_embs):
                embs.append(torch.mean(data[:seq_len[i],:], dim=0, keepdim=False))
            hidden_embs.append(torch.reshape(torch.cat(embs), (-1, 16)).detach().clone().cpu())

            infos[step * batch_size : (step+1) * batch_size, 0:3] = \
                pred.detach().clone().cpu().numpy()
            infos[step * batch_size : (step+1) * batch_size, 3] = \
                utt_ids.detach().clone().numpy()
            torch.cuda.empty_cache()
    return hidden_embs, infos
