# NEED TO: conda install tensorboard
import os
import torch
import shutil
import logging
import json 
import csv
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils import data, models, ensemble_train, ensemble_eval, split, hooks

# import path arguments
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, required=True, help='Directory to save training results')
parser.add_argument('--args', type=str, required=True, help='Path to training arguments')
parser.add_argument('--model_cat', type=str, required=False, default='finetune', help='Type of model to load')
args = parser.parse_args()


# create directory to store training results
if not os.path.isdir(args.savedir):
    os.mkdir(args.savedir)
else:
    logging.warning(f'{args.savedir} will be overwritten')
    
if not os.path.isdir(os.path.join(args.savedir,'tensorboard')):
    os.mkdir(os.path.join(args.savedir,'tensorboard'))    
    
# set up tensorboard logger
writer = SummaryWriter(log_dir=os.path.join(args.savedir,'tensorboard'))

# copy args file to training folder
shutil.copy(args.args, os.path.join(args.savedir, 'args.json'))

# read in args
savedir = args.savedir
model_cat = args.model_cat
with open(args.args) as f:
    args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
args.savedir = savedir
args.model_cat = model_cat

# check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'model will be trained on {device}')

# get initial train, val, examine splits for dataset(s)
if args.create_splits:
    logging.info('creating new split(s)')
    # create split(s)
    # TODO make general for datasets
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            split.create_init_split(args.n_train_nonmin, 0.1, 0.1, 
                                    args.n_to_examine_nonmin, args.nonmin_db_size, 
                                    'nonmin', savedir=args.savedir)
            split.create_init_split(args.n_train_min, 0.005, 0.005, 
                                    args.n_to_examine_min, args.min_db_size, 
                                    'min', savedir=args.savedir)
    else:
        split.create_init_split(args.n_train_nonmin, 0.1, 0.1, 
                                args.n_to_examine_nonmin, args.nonmin_db_size, 
                                dataset, savedir=args.savedir)
else:
    # copy initial split(s) to savedir
    logging.info(f'starting from splits in {args.splitdir}')
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            shutil.copy(os.path.join(args.splitdir, f'split_00_{dataset}.npz'), 
                        os.path.join(args.savedir, f'split_00_{dataset}.npz'))
    else: 
        shutil.copy(os.path.join(args.splitdir, f'split_00_{args.datasets}.npz'), 
                    os.path.join(args.savedir, f'split_00_{args.datasets}.npz'))

# load datasets/dataloaders
##TODO ADD FUNCTION TO TRAIN ENSEMBLES
        
    # select new samples to add from examine set
    logging.info('choosing structures to add to training set')
    # TODO make single function for quick interchange with ensemble
    n_add = 0
    add_dict={}
    for i, dataset in enumerate(args.datasets):
        ###FILL THIS IN
        
        split.create_new_split(idx_to_add, args.n_to_examine[i], 
                               al_step, dataset, 
                               savedir=args.savedir)
        
        # store sample addition info
        add_dict[dataset]=len(idx_to_add)
        
    # log sample addition info
    writer.add_scalars('samples_added', add_dict, al_step)
    
    logging.info(f'iteration {al_step} complete')
    
    n_epochs = args.n_epochs_al
        
        
        
logging.info('beginning final training epochs')

# implement early stopping
# TODO add early stopping flags to args
early_stopping = hooks.EarlyStopping(patience=10, verbose=True, 
                                     path = os.path.join(args.savedir, 'finetune_ttm_final.pt'),
                                     trace_func=logging.info)

for i in range(args.n_epochs_end):
    train_loss = train.train_energy_forces(net, train_loader, optimizer, args.energy_coeff, device)
    val_loss = train.get_pred_loss(net, val_loader, optimizer, args.energy_coeff, device)

    scheduler.step(val_loss)
    
    # log training info
    writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)
    writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)
    
    total_epochs+=1
    
    # check for stopping point
    early_stopping(valid_loss, net)
    if early_stopping.early_stop:
        break

#TODO assumes all AL steps were completed
writer.add_scalars('iteration_loss', {'train':train_loss,'val':val_loss}, args.n_al_iters+1)

# save final model, saves in early stopping
#torch.save(net.state_dict(), os.path.join(args.savedir,'finetune_ttm_final.pt'))

# close tensorboard logger
writer.close()
