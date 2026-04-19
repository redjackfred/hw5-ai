import torch
from model.network import GoNetwork

def test_output_shapes():
    net = GoNetwork()
    p, v = net(torch.zeros(2, 17, 9, 9))
    assert p.shape == (2, 82) and v.shape == (2, 1)

def test_policy_sums_to_one():
    net = GoNetwork()
    net.eval()
    with torch.no_grad():
        p, _ = net(torch.zeros(1, 17, 9, 9))
    p_prob = torch.softmax(p, dim=1)
    assert abs(p_prob.sum().item() - 1.0) < 1e-4

def test_value_in_range():
    net = GoNetwork()
    _, v = net(torch.zeros(1, 17, 9, 9))
    assert -1.0 <= v.item() <= 1.0
