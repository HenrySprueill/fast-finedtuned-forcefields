import os
import logging
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from scipy.special import erfinv


def energy_forces_loss(data, p_energies, p_forces, energy_coeff, device):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    y = torch.cat([d.y for d in data]).to(device)
    f = torch.cat([d.f for d in data]).to(device)
    energies_loss = torch.mean(torch.square(y - p_energies))
    forces_loss = torch.mean(torch.square(f - p_forces))
    total_loss = (energy_coeff)*energies_loss + (1-energy_coeff)*forces_loss
    return total_loss, energies_loss, forces_loss


def train_energy_only_single(args, model, loader, optimizer, energy_coeff, device, clip_value=150):
    """
    Loop over batches and train DataParallel model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_e_loss = []

    for data in loader:
        e = model(data)
        y = torch.cat([d.y for d in data]).to(e.device)
        e_loss = F.mse_loss(e.view(-1), y.view(-1), reduction="sum")

        with torch.no_grad():
            total_e_loss.append(e_loss.item())

        e_loss.backward()
        optimizer.step()

    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss

def train_energy_only(args, model, loader, optimizer, energy_coeff, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_e_loss = []

    for data in loader:
        e = model(data)
        y = torch.cat([d.y for d in data]).to(e.device)
        e_loss = F.mse_loss(e.view(-1), y.view(-1), reduction="sum")

        with torch.no_grad():
            total_e_loss.append(e_loss.item())

        e_loss.backward()
        optimizer.step()

    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss

def train_energy_forces_single(model, loader, optimizer, energy_coeff, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch 
    """
    model.train()
    total_ef_loss = []

    for data in loader:
        #data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()
        e = model(data)
        device = e.device
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff, e.device)
        
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())

        ef_loss.backward()
        optimizer.step()
        
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    return ave_ef_loss

def train_energy_forces(args, model, loader, optimizer, energy_coeff, device, c=0.000001, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_ef_loss = []
    total_e_loss, total_f_loss = [], []


    for data in loader:

        optimizer.zero_grad()
        e = model(data)

        concat_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        for d in concat_loader:
            d.to(e.device)
            e_tmp = model.module(d).view(-1)
            f = torch.autograd.grad(e_tmp, d.pos, grad_outputs=torch.ones_like(e_tmp), create_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff, e.device)

        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
            total_e_loss.append(e_loss.item())
            total_f_loss.append(f_loss.item())

        ef_loss.backward()
        optimizer.step()

    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    ave_f_loss = sum(total_f_loss)/len(total_f_loss)
    return ave_ef_loss, ave_e_loss, ave_f_loss



def get_error_distribution(err_list):
    """
    Compute the MAE and standard deviation of the errors in the examine set.
    """
    err_array = np.array(err_list)
    mae = np.average(np.abs(err_array))
    var = np.average(np.square(np.abs(err_array)-mae))
    return mae, np.sqrt(var)


def get_idx_to_add_single(net, examine_loader, optimizer,
                   mae, std, energy_coeff, 
                   split_file, al_step, device, min_nonmin,
                   max_to_add=0.15, error_tolerance=0.15,
                   savedir = './'):
    """
    Computes the normalized (by cluster size) errors for all entries in the examine set. It will add a max of
    max_to_add samples that are p < 0.15.
    """
    net.eval()
    all_errs = []
    for data in examine_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        energies_loss = torch.abs(data.y - e)
        f_red = torch.mean(torch.abs(data.f - f), dim=1)
        
        f_mean = torch.zeros_like(e)
        cluster_sizes = data['size'] #data.size
        for i in range(len(e)):            #loop over all clusters in batch
            energies_loss[i] /= cluster_sizes[i]
            f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])]))).clone().detach()
        
        total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
        total_err = total_err.tolist()
        all_errs += total_err
    
    with open(os.path.join(savedir, f'error_distribution_alstep{al_step}_{min_nonmin}.pkl'), 'wb') as f:
        pickle.dump(all_errs, f)    

    S = np.load(os.path.join(savedir, split_file))
    examine_idx = S["examine_idx"].tolist()
    
    cutoff = erfinv(1-error_tolerance) * std + mae
    n_samples_to_add = int(len(all_errs)*max_to_add)
    idx_highest_errors = np.argsort(np.array(all_errs))[-n_samples_to_add:]
    idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if all_errs[idx]>=cutoff]
    
    return idx_to_add

def get_idx_to_add(model, examine_loader, optimizer,
                   mae, std, energy_coeff,
                   split_file, al_step, device, min_nonmin,
                   max_to_add=0.15, error_tolerance=0.15,
                   savedir = './'):
    """
    Computes the normalized (by cluster size) errors for all entries in the examine set. It will add a max of
    max_to_add samples that are p < 0.15.
    """
    model.eval()
    all_errs = []
    for data in examine_loader:
        optimizer.zero_grad()

        e = model(data)

        concat_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        for d in concat_loader:
            d.to(e.device)
            e_tmp = model.module(d).view(-1)
            f = torch.autograd.grad(e_tmp, d.pos, grad_outputs=torch.ones_like(e_tmp), create_graph=False)[0]

        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        y = torch.cat([d.y for d in data]).to(device)
        f_true  = torch.cat([d.f for d in data]).to(device)
        energies_loss = torch.abs(y - e)
        f_red = torch.mean(torch.abs(f_true - f), dim=1)

        f_mean = torch.zeros_like(e)
        cluster_sizes = data['size'] #data.size
        for i in range(len(e)):            #loop over all clusters in batch
            energies_loss[i] /= cluster_sizes[i]
            f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])]))).clone().detach()

        total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
        total_err = total_err.tolist()
        all_errs += total_err

    with open(os.path.join(savedir, f'error_distribution_alstep{al_step}_{min_nonmin}.pkl'), 'wb') as f:
        pickle.dump(all_errs, f)

    S = np.load(os.path.join(savedir, split_file))
    examine_idx = S["examine_idx"].tolist()

    cutoff = erfinv(1-error_tolerance) * std + mae
    n_samples_to_add = int(len(all_errs)*max_to_add)
    idx_highest_errors = np.argsort(np.array(all_errs))[-n_samples_to_add:]
    idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if all_errs[idx]>=cutoff]

    return idx_to_add


def get_pred_loss_single(model, loader, optimizer, energy_coeff, device, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []
    
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff, e.device)
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            energies_loss = torch.abs(data.y - e)
            f_red = torch.mean(torch.abs(data.f - f), dim=1)

            f_mean = torch.zeros_like(e)
            cluster_sizes = data['size'] #data.size
            for i in range(len(e)):            #loop over all clusters in batch
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err
    
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    
    if val == False:
        return ave_ef_loss
    
    else:
        mae, stdvae = get_error_distribution(all_errs) #MAE and STD from EXAMINE SET
        return ave_ef_loss, mae, stdvae


def get_pred_loss(args, model, loader, optimizer, energy_coeff, device, c=0.00001, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []

    for data in loader:
        optimizer.zero_grad()

        e = model(data)
        concat_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        for d in concat_loader:
            d.to(e.device)
            e_tmp = model.module(d).view(-1)
            f = torch.autograd.grad(e_tmp, d.pos, grad_outputs=torch.ones_like(e_tmp), create_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(data, e, f, energy_coeff, e.device)

        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            y = torch.cat([d.y for d in data]).to(device)
            f_true = torch.cat([d.f for d in data]).to(device)
            energies_loss = torch.abs(y - e)
            f_red = torch.mean(torch.abs(f_true - f), dim=1)

            f_mean = torch.zeros_like(e)
            cluster_sizes = [d['size'] for d in data] 
            for i in range(len(e)):            #loop over all clusters in batch
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err

    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)

    if val == False:
        return ave_ef_loss

    else:
        mae, stdvae = get_error_distribution(all_errs) #MAE and STD from EXAMINE SET
        return ave_ef_loss, mae, stdvae

def get_pred_eloss_parallel(args, model, loader, optimizer, energy_coeff, device):
    model.eval()
    total_e_loss = []

    for data in loader:
        optimizer.zero_grad()

        e = model(data)
        y = torch.cat([d.y for d in data]).to(e.device)
        e_loss = torch.mean(torch.square(y - e))

        with torch.no_grad():
            total_e_loss.append(e_loss.item())


    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss
