"""MONAI matched-training models. Inference adapter is experimental pending lab validation."""


def transform_recipe(model):
    if model == '3dunet': return ('LoadImaged','EnsureChannelFirstd','ScaleIntensityRanged','Resized','EnsureTyped')
    if model in ('rasnet','segresnet','vnet'): return ('LoadImaged','EnsureChannelFirstd','CropForegroundd','Spacingd','ScaleIntensityRanged','EnsureTyped')
    raise ValueError(model)


def model_constructor_args(model):
    common={'spatial_dims':3,'in_channels':1}
    if model in ('rasnet','segresnet'):
        return dict(common,out_channels=2,init_filters=16,dropout_prob=.1)
    if model=='vnet': return dict(common,out_channels=2,act='elu',dropout_prob_down=.1,dropout_prob_up=(.1,.1),bias=True)
    if model=='3dunet': return dict(common,out_channels=1,channels=(16,32,64,128,256),strides=(2,2,2,2),num_res_units=2,norm='instance',dropout=.15)
    raise ValueError(model)


def build_model(name, checkpoint, device):
    import torch
    from monai.networks.nets import SegResNet, VNet, UNet
    from Phase3_Local_Integration.rasnet_model import RASNet
    from pathlib import Path
    if not Path(checkpoint).is_file(): raise FileNotFoundError(checkpoint)
    cls={'rasnet':RASNet,'segresnet':SegResNet,'vnet':VNet,'3dunet':UNet}[name]
    net=cls(**model_constructor_args(name)).to(device)
    obj=torch.load(checkpoint,map_location=device,weights_only=True)
    weights=obj['model_state_dict'] if name=='3dunet' else obj
    net.load_state_dict(weights, strict=True)
    return net.eval()


def build_transform(name):
    from monai import transforms as t
    steps=[t.LoadImaged(keys=['image']),t.EnsureChannelFirstd(keys=['image'])]
    if name != '3dunet':
        steps.extend([t.CropForegroundd(keys=['image'],source_key='image'),t.Spacingd(keys=['image'],pixdim=(.5,.5,.5),mode='bilinear')])
    steps.append(t.ScaleIntensityRanged(keys=['image'],a_min=-100,a_max=800,b_min=0,b_max=1,clip=True))
    if name=='3dunet': steps.append(t.Resized(keys=['image'],spatial_size=(128,128,128),mode='trilinear'))
    steps.append(t.EnsureTyped(keys=['image'],track_meta=True))
    return t.Compose(steps)


def infer_probability(name, net, image_path, device, roi_size=(96,96,96), sw_batch_size=1, tta=False):
    """Return foreground probabilities in the original SimpleITK z,y,x grid."""
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd
    if hasattr(net, 'eval'):
        net.eval()
    if name == 'vnet':
        roi_size = tuple(((int(r) + 15) // 16) * 16 for r in roi_size)
    transform=build_transform(name)
    batch=transform({'image':image_path})
    x=batch['image'].unsqueeze(0).to(device)
    def predict(tensor):
        out=net(tensor) if name=='3dunet' else sliding_window_inference(tensor,roi_size,sw_batch_size,net,overlap=.25)
        if isinstance(out,(list,tuple)): out=out[0]
        return torch.sigmoid(out) if name=='3dunet' else torch.softmax(out,dim=1)[:,1:2]
    with torch.no_grad():
        p=predict(x)
        if tta:
            for axis in (2,3,4): p += torch.flip(predict(torch.flip(x,(axis,))),(axis,))
            p /= 4
    batch['pred']=p.squeeze(0)
    inverted=Invertd(keys=['pred'],transform=transform,orig_keys=['image'],nearest_interp=False,to_tensor=True)(batch)['pred']
    return np.transpose(inverted.squeeze(0).detach().cpu().numpy(),(2,1,0)).astype(np.float32, copy=False)


def build_random_model(name):
    """Construct the exact architecture, untrained, for synthetic shape checks only."""
    from monai.networks.nets import SegResNet, VNet, UNet
    from Phase3_Local_Integration.rasnet_model import RASNet
    cls={'rasnet':RASNet,'segresnet':SegResNet,'vnet':VNet,'3dunet':UNet}[name]
    return cls(**model_constructor_args(name)).eval()
