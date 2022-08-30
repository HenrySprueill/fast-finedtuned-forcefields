import os 
import torch
import numpy as np
import ensemble_eval
import pickle



def train_net(args, 
              net,
              al_step,
              device,
              examine_loaders = None,
              n_epochs = None):
    
    train_loader, val_loader, examine_loaders = data.bulk_dataloader(args, split=str(al_step+1).zfill(2))

    # load model
    net = models.load_model(args, args.model_cat, device=device)
    logging.info(f'model loaded from {args.start_model}')

    #initialize optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8, min_lr=0.000001)


    # train
    if n_epochs is None:
        n_epochs = args.n_epochs
    total_epochs = 0

    logging.info('beginning training...')
    #while number_added > args.al_threshold:
    
    logging.info(f'beginning active learning interation {al_step}')
    for e in range(n_epochs):
        # train model
        train_loss = train.train_energy_forces(net, train_loader, optimizer, args.energy_coeff, device)

        # get validation set loss
        val_loss = train.get_pred_loss(net, val_loader, optimizer, args.energy_coeff, device)

        scheduler.step(val_loss)

        # log training info
        writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)

        # on same plot
        writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)

        total_epochs+=1

        # Save current model
    if args.save_models:
        torch.save(net.state_dict(), os.path.join(args.savedir, f'finetune_ttm_alstep{al_step}.pt'))

    # on same plot
    writer.add_scalars('iteration_loss', {'train':train_loss,'val':val_loss}, al_step)
    
    if examine_loaders is not None:
        predictions = []
        for i, dataset in enumerate(args.datasets):
            predictions += get_predictions(net, examine_loaders[i], device)
       
    return predictions