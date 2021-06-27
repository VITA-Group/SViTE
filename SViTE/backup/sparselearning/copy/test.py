import torch

grad = torch.tensor([[-0.1,-0.2,-0.3],[0,0,0],[0.1,0.2,0.3]])
new_mask = torch.tensor([[0,0,0],[0,0,0],[0,0,0]])
print(new_mask)
total_regrowth = 6
y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
print(new_mask)
