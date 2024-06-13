import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from model.GRU_TFM import GRU_TFM_reg_mlt
import model.hparams as hp
from model.train import run, train, fair_train, emb_run
from interspeech.utils.sampler import SplitDistSampler
from loader.data_loader import MSP_Dataset, seq_collate_pad_zero
import utils.metric, utils.loss
from tqdm import tqdm
from pathlib import Path
import numpy as np
import joblib
import argparse
import os

parser = argparse.ArgumentParser(description='GRU_TFM SLT model training')
parser.add_argument('--exp_name', default='exp/new_exp',
                    help='exp folder for save')
parser.add_argument('--target_name', default='A',
                    help='learning target, \'A\', \'V\', or \'D\'')
parser.add_argument('--lam', default=1.0,
                    help='scale of the penalites')
args = parser.parse_args()
path = Path(args.exp_name+'/')

# parameters
DEVICE = 0
LR = 1e-5
EPOCH = 200
BATCH_SIZE = 128
PATIENCE = 20
MIN_SAMPLE = 32
NO_UNKNOWN = True
TOLERANCE = 0.05
SEED = 3600
SPLIT = 4
INITIAL_STEP = 1  # set to EPOCH to run baseline model
LAM = float(args.lam)
#PSEUDO_LABEL = 'hdb_label'
PSEUDO_LABEL = None
FEATURE = 'hubert'
PROTECTED = 'spk_id'


#%% environment setup
torch.cuda.set_device(DEVICE)
device = torch.device("cuda", DEVICE)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.manual_seed(3600)

# clear save folder
path.mkdir(parents=True, exist_ok=True)
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    try: 
        if os.path.isfile(file_path or os.path.islink(file_path)):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
            
# path to your meta file
train_fea_meta_file = './feature_extract/MSP/train_meta_hdb.csv'
valid_fea_meta_file = './feature_extract/MSP/valid_meta_hdb.csv'
test_fea_meta_file = './feature_extract/MSP/test_meta_hdb.csv'

# model setup
model = GRU_TFM_reg_mlt(hp.FEAT_DIM, hp.HIDDEN_DIM, hp.HIDDEN_LAYERS_NUM, hp.TFM_HEAD,\
                        max_length = 2500, num_task=3, dropout_r = hp.DROPOUT_R)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
fairness = utils.loss.Discrimination(device=device)
criterion = utils.loss.CCCLoss()

fairness_eval = utils.metric.Discrimination(min_sample=MIN_SAMPLE)

# load dataset
train_dataset = MSP_Dataset(train_fea_meta_file, target_name = args.target_name, feature_name=FEATURE, protect=PROTECTED, min_sample=MIN_SAMPLE, ignore_unknown=NO_UNKNOWN, pseudo_label=PSEUDO_LABEL)
train_sampler = SplitDistSampler(train_dataset, train_dataset.getProtect(), split=SPLIT, batch_size=BATCH_SIZE, seed=SEED)
balance_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, \
    collate_fn = seq_collate_pad_zero, sampler = train_sampler, shuffle = False, drop_last=True, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, \
    collate_fn = seq_collate_pad_zero, shuffle = True, drop_last=True, num_workers=1)

emb_preprocess = train_dataset.getPreprocessor()

valid_dataset = MSP_Dataset(valid_fea_meta_file, target_name = args.target_name, feature_name=FEATURE, pseudo_label=PSEUDO_LABEL)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, \
    collate_fn = seq_collate_pad_zero, shuffle = False, \
    drop_last=False, num_workers=1)

# result record
RE_BOX = {  
    'epoch_loss': [],
    'best_epoch': 0,
    'gender_fair_epoch': 0,
    'spk_id_fair_epoch': 0,
    'clu_id_fair_epoch': 0,
    'performance': [],
    'gender_fair': [],
    'spk_id_fair': [],
    'clu_id_fair': [],
    'train': {
        'gender': None,
        'length': None,
        'spk_id': None,
        'clu_id': None,
        'values': None,
        'metric': None,
        'pred': None,
    },
    'valid': {
        'gender': None,
        'length': None,
        'spk_id': None,
        'clu_id': None,
        'values': None,
        'metric': None,
        'pred': None,
    },
    'test': {
        'gender': None,
        'length': None,
        'spk_id': None,
        'clu_id': None,
        'values': None,
        'metric': None,
        'pred': None,
    }
}

RE_BOX['train']['spk_id'] = np.array(train_dataset.getSPK_ID())
RE_BOX['train']['gender'] = np.array(train_dataset.getGender())
RE_BOX['train']['clu_id'] = np.array(train_dataset.getCLU_ID())
RE_BOX['train']['length'] = np.array(train_dataset.getLength())
RE_BOX['train']['values'] = np.array(train_dataset.getValues())
RE_BOX['train']['metric'] = np.empty((EPOCH, 6, 3))
RE_BOX['train']['pred'] = np.empty((EPOCH, len(train_dataset), 3), dtype=np.single)
RE_BOX['train']['pred'][:] = np.nan

RE_BOX['valid']['spk_id'] = np.array(valid_dataset.getSPK_ID())
RE_BOX['valid']['gender'] = np.array(valid_dataset.getGender())
RE_BOX['valid']['length'] = np.array(valid_dataset.getLength())
RE_BOX['valid']['clu_id'] = np.array(valid_dataset.getCLU_ID())
RE_BOX['valid']['values'] = np.array(valid_dataset.getValues())
RE_BOX['valid']['metric'] = np.empty((EPOCH, 6, 3))
RE_BOX['valid']['pred'] = np.empty((EPOCH, len(valid_dataset), 3), dtype=np.single)
RE_BOX['valid']['pred'][:] = np.nan

#%% training and validation
pbar = tqdm(range(EPOCH), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

early_stop = 0
for epoch in pbar:
    # train
    train_sampler.set_epoch(epoch)
    if epoch < INITIAL_STEP:
        whole_infos, whole_loss = train(model, train_loader, optimizer,
                                            criterion, device, BATCH_SIZE)
    else:
        whole_infos, whole_loss = fair_train(model, balance_loader, optimizer,
                                            criterion, fairness, device, BATCH_SIZE, LAM)

    RE_BOX['epoch_loss'].append(whole_loss)
    RE_BOX['train']['pred'][epoch, whole_infos[:, 3].astype(int), :] = \
        whole_infos[:, :3]
    pred = RE_BOX['train']['pred'][epoch]
    true = RE_BOX['train']['values'][~np.isnan(pred).any(axis=1)]
    gender = RE_BOX['train']['gender'][~np.isnan(pred).any(axis=1)]
    spk_id = RE_BOX['train']['spk_id'][~np.isnan(pred).any(axis=1)]
    clu_id = RE_BOX['train']['clu_id'][~np.isnan(pred).any(axis=1)]
    pred = pred[~np.isnan(pred).any(axis=1)]
    
    # evaluation of training dataset
    RE_BOX['train']['metric'][epoch, 0] = utils.metric.PCC(pred, true)
    RE_BOX['train']['metric'][epoch, 1] = utils.metric.CCC(pred, true)
    RE_BOX['train']['metric'][epoch, 2] = utils.metric.SPC(pred, true)
    RE_BOX['train']['metric'][epoch, 3] = fairness_eval(pred, gender, MIN_SAMPLE)
    RE_BOX['train']['metric'][epoch, 4] = fairness_eval(pred, spk_id, MIN_SAMPLE)
    RE_BOX['train']['metric'][epoch, 5] = fairness_eval(pred, clu_id, MIN_SAMPLE)

    
    # validation
    whole_infos = run(model, valid_loader, device, BATCH_SIZE, len(valid_dataset))

    RE_BOX['valid']['pred'][epoch, whole_infos[:, 3].astype(int), :] = \
        whole_infos[:, :3]
    pred = RE_BOX['valid']['pred'][epoch]
    true = RE_BOX['valid']['values'][~np.isnan(pred).any(axis=1)]
    gender = RE_BOX['valid']['gender'][~np.isnan(pred).any(axis=1)]
    spk_id = RE_BOX['valid']['spk_id'][~np.isnan(pred).any(axis=1)]
    clu_id = RE_BOX['valid']['clu_id'][~np.isnan(pred).any(axis=1)]
    pred = pred[~np.isnan(pred).any(axis=1)]
    pred = np.clip(pred, -1.0, 1.0)
    # evaluation of training dataset
    RE_BOX['valid']['metric'][epoch, 0] = utils.metric.PCC(pred, true)
    RE_BOX['valid']['metric'][epoch, 1] = utils.metric.CCC(pred, true)
    RE_BOX['valid']['metric'][epoch, 2] = utils.metric.SPC(pred, true)
    RE_BOX['valid']['metric'][epoch, 3] = fairness_eval(pred, gender, MIN_SAMPLE)
    RE_BOX['valid']['metric'][epoch, 4] = fairness_eval(pred, spk_id, MIN_SAMPLE)
    RE_BOX['valid']['metric'][epoch, 5] = fairness_eval(pred, clu_id, MIN_SAMPLE)

    performance = np.mean(RE_BOX['valid']['metric'][epoch, 1])
    gender_fair = np.mean(RE_BOX['valid']['metric'][epoch, 5])
    clu_id_fair = np.mean(RE_BOX['valid']['metric'][epoch, 6])
    spk_id_fair = np.mean(RE_BOX['valid']['metric'][epoch, 7])
    RE_BOX['performance'].append(performance)
    RE_BOX['gender_fair'].append(gender_fair)
    RE_BOX['spk_id_fair'].append(spk_id_fair)
    RE_BOX['clu_id_fair'].append(clu_id_fair)

    if epoch == 0 or performance > RE_BOX['performance'][RE_BOX['best_epoch']]:
        RE_BOX['best_epoch'] = epoch
        RE_BOX['gender_fair_epoch'] = epoch
        RE_BOX['spk_id_fair_epoch'] = epoch
        RE_BOX['clu_id_fair_epoch'] = epoch
        torch.save(model, args.exp_name+'/best_va_result.pt')
        torch.save(model, args.exp_name+'/gender_fair_va_result.pt')
        torch.save(model, args.exp_name+'/spk_id_fair_va_result.pt')
        torch.save(model, args.exp_name+'/clu_id_fair_va_result.pt')
        joblib.dump(RE_BOX, args.exp_name+'/info_ckpt.pkl')
        with open(args.exp_name+'/epoch_'+str(epoch).zfill(3)+'.best', 'w') as fp:
            pass
        pbar.set_description('CCC({:}): {:.3g}, gender_fair({:}): {:.3g}, spk_fair({:}): {:.3g}, clu_fair({:}): {:.3g}'.format(\
            epoch, performance, epoch, gender_fair, epoch, spk_id_fair, epoch, clu_id_fair
        )) 
        early_stop = 0
    elif (performance >= RE_BOX['performance'][RE_BOX['best_epoch']] * (1-TOLERANCE)):
        if gender_fair < RE_BOX['gender_fair'][RE_BOX['gender_fair_epoch']]:
            RE_BOX['gender_epoch'] = epoch
            torch.save(model, args.exp_name+'/gender_fair_va_result.pt')
            with open(args.exp_name+'/epoch_'+str(epoch).zfill(3)+'.gender_fair', 'w') as fp:
                pass
        if spk_id_fair < RE_BOX['spk_id_fair'][RE_BOX['spk_id_fair_epoch']]:
            RE_BOX['spk_id_epoch'] = epoch
            torch.save(model, args.exp_name+'/spk_id_fair_va_result.pt')
            with open(args.exp_name+'/epoch_'+str(epoch).zfill(3)+'.spk_id_fair', 'w') as fp:
                pass
        if clu_id_fair < RE_BOX['clu_id_fair'][RE_BOX['clu_id_fair_epoch']]:
            RE_BOX['clu_id_epoch'] = epoch
            torch.save(model, args.exp_name+'/clu_id_fair_va_result.pt')
            with open(args.exp_name+'/epoch_'+str(epoch).zfill(3)+'.clu_id_fair', 'w') as fp:
                pass
        joblib.dump(RE_BOX, args.exp_name+'/info_ckpt.pkl')
        pbar.set_description('CCC({:}): {:.3g}, mp_fair({:}): {:.3g}, ks_fair({:}): {:.3g}, ws_fair({:}): {:.3g}'.format(\
            RE_BOX['best_epoch'], RE_BOX['performance'][RE_BOX['best_epoch']], 
            RE_BOX['gender_fair_epoch'], RE_BOX['gender_fair'][RE_BOX['gender_fair_epoch']], 
            RE_BOX['spk_id_fair_epoch'], RE_BOX['spk_id_fair'][RE_BOX['spk_id_fair_epoch']], 
            RE_BOX['clu_id_fair_epoch'], RE_BOX['clu_id_fair'][RE_BOX['clu_id_fair_epoch']], 
        ))
        early_stop += 1
    else:
        early_stop += 1
    
#%% testing
    model = torch.load(args.exp_name+'/best_va_result.pt')
    model.to(device)
    test_dataset = MSP_Dataset(test_fea_meta_file, target_name = args.target_name, feature_name=FEATURE, pseudo_label=PSEUDO_LABEL)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, \
        collate_fn = seq_collate_pad_zero, shuffle = False, \
        drop_last=False, num_workers=1)
    
    RE_BOX['test']['spk_id'] = np.array(test_dataset.getSPK_ID())
    RE_BOX['test']['gender'] = np.array(test_dataset.getGender())
    RE_BOX['test']['length'] = np.array(test_dataset.getLength())
    RE_BOX['test']['clu_id'] = np.array(test_dataset.getCLU_ID())
    RE_BOX['test']['values'] = np.array(test_dataset.getValues())
    RE_BOX['test']['metric'] = np.empty((1, 6, 3))
    RE_BOX['test']['utt_id'] = np.array(test_dataset.getUTT_ID())
    RE_BOX['test']['pred'] = np.empty((EPOCH, len(test_dataset), 3), dtype=np.single)
    RE_BOX['test']['pred'][:] = np.nan
    hidden_embs, whole_infos = emb_run(model, test_loader, device, BATCH_SIZE, len(test_dataset))
    embs = torch.cat(hidden_embs)[whole_infos[:, 3].astype(int), :].numpy()
    joblib.dump(embs, args.exp_name + '/emb.pkl')
    RE_BOX['test']['pred'][0, whole_infos[:, 3].astype(int), :] = \
        whole_infos[:, :3]
    pred = RE_BOX['test']['pred'][0]
    true = RE_BOX['test']['values'][~np.isnan(pred).any(axis=1)]
    gender = RE_BOX['test']['gender'][~np.isnan(pred).any(axis=1)]
    spk_id = RE_BOX['test']['spk_id'][~np.isnan(pred).any(axis=1)]
    clu_id = RE_BOX['test']['clu_id'][~np.isnan(pred).any(axis=1)]
    pred = pred[~np.isnan(pred).any(axis=1)]
    pred = np.clip(pred, -1.0, 1.0)
    # evaluation of training dataset
    RE_BOX['test']['metric'][0, 0] = utils.metric.PCC(pred, true)
    RE_BOX['test']['metric'][0, 1] = utils.metric.CCC(pred, true)
    RE_BOX['test']['metric'][0, 2] = utils.metric.SPC(pred, true)
    RE_BOX['test']['metric'][0, 3] = utils.metric.mean_parity_diff(pred, gender, MIN_SAMPLE)
    RE_BOX['test']['metric'][0, 4] = utils.metric.mean_parity_diff(pred, spk_id, MIN_SAMPLE)
    RE_BOX['test']['metric'][0, 5] = utils.metric.mean_parity_diff(pred, clu_id, MIN_SAMPLE)
    joblib.dump(RE_BOX, args.exp_name+'/info_ckpt.pkl')
    performance = np.mean(RE_BOX['test']['metric'][0, 1])
    mp_fairness = np.mean(RE_BOX['test']['metric'][0, 5])
    ks_fairness = np.mean(RE_BOX['test']['metric'][0, 6])
    ws_fairness = np.mean(RE_BOX['test']['metric'][0, 7])
    print('CCC: {:.3g}, gender_fair: {:.3g}, speaker_fair: {:.3g}, cluster_fair: {:.3g}'.format(\
        performance, mp_fairness, ks_fairness, ws_fairness
    ))
