import torch.optim
import pdb

def build_optimizer(model, optim_cfg):
    assert "type" in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop("type")
    optim = getattr(torch.optim, optim_type)

    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)

def build_optimizer_wo_enc(model, optim_cfg):
    assert "type" in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop("type")
    optim = getattr(torch.optim, optim_type)
    #pdb.set_trace()
    #list(model.inst_mask_head_angular.parameters())
    #optim(model.parameters(), **_optim_cfg)
    #return optim(filter(lambda p: p.requires_grad, model.inst_mask_head.parameters()), **_optim_cfg)

    #return optim(filter(lambda p: p.requires_grad, model.inst_mask_head_angular.parameters()), **_optim_cfg)
    #pdb.set_trace()
    optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
    
    for name, param in model.named_parameters():
        if 'input_conv' in name or 'unet' in name or 'output_layer' in name:
            param.requires_grad = False
            print(name)
        else:
            param.requires_grad = True
    
    return optimizer
def build_optimizer_tune(model, optim_cfg):
    assert "type" in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop("type")
    optim = getattr(torch.optim, optim_type)
    #pdb.set_trace()
    #list(model.inst_mask_head_angular.parameters())
    #optim(model.parameters(), **_optim_cfg)
    #return optim(filter(lambda p: p.requires_grad, model.inst_mask_head.parameters()), **_optim_cfg)

    #return optim(filter(lambda p: p.requires_grad, model.inst_mask_head_angular.parameters()), **_optim_cfg)
    #pdb.set_trace()
    optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
    for name, param in model.named_parameters():
        if 'inst_mask_head_angular' in name or 'tower' in name:
            param.requires_grad = True
            print(name)
        else:
            param.requires_grad = False
   
    return optimizer