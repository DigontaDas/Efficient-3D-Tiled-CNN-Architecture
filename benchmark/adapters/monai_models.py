"""MONAI matched-training models and unified native-grid probability inference."""


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


def infer_monai_both_passes(
    name: str,
    net,
    image_path: str,
    device: str,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 8,
    need_tta: bool = False,
    transform=None,
    batch: dict | None = None,
    use_amp: bool | None = None,
):
    """
    Unified MONAI preprocessing + inference returning (prob_1pass_zyx, prob_tta_zyx or None)
    in the original SimpleITK (Z, Y, X) grid. Uses CUDA autocast when device is CUDA
    (matching the training validation loops in train_*_200ep.py) unless use_amp is overridden.
    """
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import Invertd

    if hasattr(net, 'eval'):
        net.eval()
    if name == 'vnet':
        roi_size = tuple(((int(r) + 15) // 16) * 16 for r in roi_size)

    if transform is None:
        transform = build_transform(name)
    if batch is None:
        batch = transform({'image': str(image_path)})
    x = batch['image'].unsqueeze(0).to(device)

    amp_enabled = str(device).startswith('cuda') if use_amp is None else bool(use_amp)

    def predict(tensor):
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            out = (
                net(tensor)
                if name == '3dunet'
                else sliding_window_inference(
                    tensor, roi_size, sw_batch_size, net, overlap=0.25
                )
            )
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = (
                torch.sigmoid(out.float())
                if name == '3dunet'
                else torch.softmax(out.float(), dim=1)[:, 1:2]
            )
        return probs.float()

    with torch.no_grad():
        p_raw = predict(x)
        p_tta = None
        if need_tta:
            acc = p_raw.clone()
            for axis in (2, 3, 4):
                acc += torch.flip(predict(torch.flip(x, (axis,))), (axis,))
            p_tta = acc / 4.0

    batch_raw = dict(batch)
    batch_raw['pred'] = p_raw.squeeze(0)
    inv_raw = Invertd(
        keys=['pred'],
        transform=transform,
        orig_keys=['image'],
        nearest_interp=False,
        to_tensor=True,
    )(batch_raw)['pred']
    prob_1pass = np.transpose(
        inv_raw.squeeze(0).detach().cpu().numpy(), (2, 1, 0)
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(prob_1pass)):
        raise RuntimeError(f'Non-finite probability output from {name} (1-pass) on {image_path}')

    prob_tta_zyx = None
    if p_tta is not None:
        batch_tta = dict(batch)
        batch_tta['pred'] = p_tta.squeeze(0)
        inv_tta = Invertd(
            keys=['pred'],
            transform=transform,
            orig_keys=['image'],
            nearest_interp=False,
            to_tensor=True,
        )(batch_tta)['pred']
        prob_tta_zyx = np.transpose(
            inv_tta.squeeze(0).detach().cpu().numpy(), (2, 1, 0)
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(prob_tta_zyx)):
            raise RuntimeError(f'Non-finite probability output from {name} (TTA) on {image_path}')

    return prob_1pass, prob_tta_zyx


def infer_probability(name, net, image_path, device, roi_size=(96,96,96), sw_batch_size=1, tta=False, use_amp=None):
    """Return foreground probabilities in the original SimpleITK z,y,x grid via unified infer_monai_both_passes."""
    prob_1pass, prob_tta = infer_monai_both_passes(
        name=name,
        net=net,
        image_path=image_path,
        device=device,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        need_tta=bool(tta),
        use_amp=use_amp,
    )
    return prob_tta if tta else prob_1pass


def build_random_model(name):
    """Construct the exact architecture, untrained, for synthetic shape checks only."""
    from monai.networks.nets import SegResNet, VNet, UNet
    from Phase3_Local_Integration.rasnet_model import RASNet
    cls={'rasnet':RASNet,'segresnet':SegResNet,'vnet':VNet,'3dunet':UNet}[name]
    return cls(**model_constructor_args(name)).eval()

