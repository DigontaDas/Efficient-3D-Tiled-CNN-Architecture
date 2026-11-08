import torch
import pytest
from benchmark.adapters.monai_models import build_random_model

@pytest.mark.parametrize('name',['rasnet','segresnet','vnet','3dunet'])
def test_random_architecture_output_grid(name):
    torch.set_num_threads(1)
    net=build_random_model(name).eval()
    with torch.no_grad(): out=net(torch.zeros((1,1,32,32,32)))
    if isinstance(out,(tuple,list)): out=out[0]
    assert out.shape[2:]==(32,32,32)
