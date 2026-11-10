from benchmark.adapters.monai_models import transform_recipe, model_constructor_args

def test_training_validation_transform_contract():
    for model in ('rasnet','segresnet','vnet'):
        assert transform_recipe(model)==('LoadImaged','EnsureChannelFirstd','CropForegroundd','Spacingd','ScaleIntensityRanged','EnsureTyped')
    assert transform_recipe('3dunet')==('LoadImaged','EnsureChannelFirstd','ScaleIntensityRanged','Resized','EnsureTyped')
    assert model_constructor_args('vnet')['dropout_prob_up']==(0.1,0.1)
    assert model_constructor_args('3dunet')['out_channels']==1

def test_adapter_recipe_matches_training_source():
    import ast
    from pathlib import Path
    from benchmark.adapters.monai_models import build_transform
    base=Path('Q1_Publication_Package/matched_200ep_benchmark')
    for model in ('rasnet','segresnet','vnet','3dunet'):
        tree=ast.parse((base/f'train_{model}_200ep.py').read_text(encoding='utf-8'))
        candidates=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='val_transforms' for t in n.targets)]
        assert len(candidates)==1
        names=[x.func.id for x in candidates[0].args[0].elts if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
        # ToTensord and ConvertToPlainTensor in training are replaced solely to retain inversion metadata.
        names=[x for x in names if x not in ('ToTensord','ConvertToPlainTensor')] + ['EnsureTyped']
        assert names == [type(x).__name__ for x in build_transform(model).transforms]
