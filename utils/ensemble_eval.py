import os
import torch
import numpy as np



def compute_sufficient_statistics(data, external_values=None, axis = 0):
    vals = data
    if external_values is not None:
        vals = external_values[vals]
    n = len(vals)
    mu = np.mean(vals, axis = axis)
    s = np.std(vals, ddof=1, axis = axis)
    return n, mu, s
    
    

def compute_T_from_sufficient_statistics(stats1, stats2):
    n_1, mu_1, s_1 = stats1
    
    n_2, mu_2, s_2 = stats2
    
    s = np.sqrt(((n_1-1)*s_1**2 + (n_2-1)*s_2**2)/(n_1+n_2-2))
    
    T = (mu_1 - mu_2)/s/np.sqrt(1/n_1 + 1/n_2)
    return T



def select_samples_ensembles(examine_results,
                             train_results,
                             batch_size,
                             alpha,
                             num_resamples=10000,
                             cushion = 0,
                             limit_clusters=-1,
                             shuffle_samples = False,
                             seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    #Collapse results into 1-D arrays, taking ensemble std
    examine_arr = np.std(examine_results, ddof=1, axis = 0)
    train_arr = np.std(train_results, ddof=1, axis = 0)
    
    #Get a list of indeces of examine_arr
    examine_idx = np.array([i for i in range(len(examine_arr))])
    if shuffle_samples:
        np.random.shuffle(examine_idx)
        
    #Batch the examine_array
    residual = len(examine_idx) % batch_size
    if residual > 0:
        residual_batch = examine_idx[-(batch_size + residual):]
    batches = examine_idx[:-(batch_size + residual)].reshape(-1, batch_size)
      
    #Compute statistics for examine set (mu, s results are arrays)
    n_e, mu_e, s_e = compute_sufficient_statistics(batches, external_values=examine_arr, axis=1)
    #Compute train statistics
    n_t, mu_t, s_t = compute_sufficient_statistics(train_arr)
    #compute statistics for the residual batch
    if residual > 0:
        n_r, mu_r, s_r = compute_sufficient_statistics(residual_batch)
    #copy and stack the training data to concatenate it with examine data
    train_arr_stacked = np.tile(train_arr, (batches.shape[0], 1))
    
    #Compute the two-sample T statistics 
    Ts = compute_T_from_sufficient_statistics((n_e, mu_e, s_e), (n_t, mu_t, s_t))
    
    if residual > 0:
        
        n_r, mu_r, s_r = compute_sufficient_statistics(residual_batch, external_values=examine_arr)
        Tr = compute_T_from_sufficient_statistics((n_r, mu_r, s_r), (n_t, mu_t, s_t))
    
    counts = np.zeros_like(mu_e)
    count_residuals = 0
    for b in range(num_resamples): #take resamples
        
        resampling_data = np.concatenate((train_arr_stacked, examine_arr[batches]), axis=1)
        np.random.shuffle(resampling_data) #shuffle the data
        
        train_resample_data = resampling_data[:,:n_t]#get simulated train resample data
        n_rt, mu_rt, s_rt = compute_sufficient_statistics(train_resample_data, 
                                                          axis=1)
        examine_resample_data = resampling_data[:,n_t:]#get simulated examine resample data
        n_re, mu_re, s_re = compute_sufficient_statistics(examine_resample_data, 
                                                          axis=1)
        #compute the resampling T
        T_resample = compute_T_from_sufficient_statistics((n_rt, mu_rt, s_rt), (n_re, mu_re, s_re))
        #Is this as extreme as our test statistic?
        counts += np.greater_equal(T_resample, Ts-cushion)
        
        if residual > 0:#same thing for residual array
            residual_data = np.concatenate((train_arr, examine_arr[residual_batch]))
            
            np.random.shuffle(residual_data)
            train_residual_data = residual_data[:n_t]
            n_rt, mu_rt, s_rt = compute_sufficient_statistics(train_residual_data, 
                                                          axis=0)
            examine_residual_data = residual_data[n_t:]
            n_re, mu_re, s_re = compute_sufficient_statistics(examine_residual_data, 
                                                          axis=0)
            T_residual = compute_T_from_sufficient_statistics((n_rt, mu_rt, s_rt), (n_re, mu_re, s_re))

            if T_residual >= Tr-cushion:
                count_residuals+=1
        
    #Compute portion of resamples that were as extreme as the test statistic (p-value)
    p_resample = counts/num_resamples
    p_rejections = np.less_equal(p_resample, alpha)#check alpha
    
    samples_to_keep = []
    for i, batch in enumerate(batches):
        if p_rejections[i]:#batch rejects H_0
            batch = batches[i]
            #filter for samples higher than the mean
            samples_to_keep += [idx for idx in batch if examine_arr[idx] > np.mean(examine_arr[batch])]
    
    if residual > 0:#same thing for residuals
        p_residual = count_residuals/num_resamples
        if p_residual <= alpha:
            samples_to_keep += [idx for idx in residual_batch if examine_arr[idx] > np.mean(examine_arr[residual_batch])]
    return samples_to_keep#returns indeces of samples chosen to include in training



def get_predictions(model, loader, device):
    """
    Gets the model predictions on a data loader
    """
    model.eval()
    all_vals = []
    
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        all_valls += e.tolist()

    return all_vals