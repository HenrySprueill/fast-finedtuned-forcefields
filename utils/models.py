import torch
import torch.nn as nn
import sys
import logging
import argparse

def load_model(args, model_cat, mode='eval', device='cpu', frozen=False):
    """
    Load trained model for eval
    model_cat = ['ipu', 'finetune', 'multifi']
    """
    
    if model_cat in ['ipu','finetune']:
        net = load_pretrained_model(args, model_cat, device=device, frozen=frozen)
    else:
        sys.path.insert(0, '../schnet')

        from model import SchNet
        
        net = MultiFiSchNet(args, device = device)
    
    if mode=='eval':
        # set to eval mode
        net.eval()
        print('model set to eval')

    return net

# DONT CHANGE THIS ONE
def load_pretrained_model(args, model_cat='', device='cpu', frozen=False):
    """
    Load single SchNet model
    """
    device = torch.device(device)
    
    # load codebase depending on model_cat
    if model_cat == 'ipu':
        sys.path.insert(0, '../schnet-ipu')
    else:
        sys.path.insert(0, '../schnet')

    from model import SchNet
    
    # load state dict of trained model
    state=torch.load(args.start_model)
    
    # extract model params from model state dict
    num_gaussians = state['basis_expansion.offset'].shape[0]
    num_filters = state['interactions.0.mlp.0.weight'].shape[0]
    num_interactions = len([key for key in state.keys() if '.lin.bias' in key])
    
    # load model architecture
    net = SchNet(num_features = num_filters,
                 num_interactions = num_interactions,
                 num_gaussians = num_gaussians,
                 cutoff = 6.0)
    
    logging.info(f'model loaded from {args.start_model}')
    
    if args.load_state:
        # load trained weights into model
        net.load_state_dict(state)
        logging.info('model weights loaded')
    
    net.to(device)
    
    #register backward hook --> gradient clipping
    if not frozen:
        for p in net.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))

    return net


class MultiFiSchNet(torch.nn.Module):
    def __init__(self,
                 args,
                 device = 'cpu'):
        """
        :param model_path (str): Path to trained model
        :param device (str): Device to run model on ['cpu', 'cuda']
        """
        super().__init__()
        
        sys.path.insert(0, '/people/herm379/exalearn/IPU_trained_models/pnnl_sandbox/schnet')

        from model import SchNet
        
        self.args = args
        
        # load pretrained model
        self.lowfi_model = load_pretrained_model(args, model_cat='finetune', frozen=True, device=device)

        # freeze lowfi model layers
        for param in self.lowfi_model.parameters():
            param.requires_grad = False
            
        # load empty model with smaller architecture
        state=torch.load(args.start_model, map_location=torch.device(device))
        num_gaussians = state['basis_expansion.offset'].shape[0]
        num_filters = state['interactions.0.mlp.0.weight'].shape[0]
        num_interactions = len([key for key in state.keys() if '.lin.bias' in key])
        
        self.dif_model = SchNet(num_features = int(num_filters/2),
                     num_interactions = int((num_interactions+1)/2),
                     num_gaussians = num_gaussians,
                     cutoff = 6.0)
        
        self.dif_model.to(device)
        
        # correlation/sum layer 
        self.correlation = nn.Linear(1, 1, bias=False, device=device)
        
        for p in self.correlation.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))


    def forward(self, data):
        """
        Forward pass of the SchNet model
        :param data: data from data loader
        """     
        
        y_low = self.correlation(self.lowfi_model(data).view(-1,1)).T[0]
        y_dif = self.dif_model(data)
        
        # interleave y_low and y_dif
        y = torch.stack((y_low,y_dif), dim=1).view(-1,2)
        y = torch.sum(y, dim=1)
        
        return y
        
