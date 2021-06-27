import torch 




all_masks = {}
for i in range(8):
    all_masks[i] = torch.load('{}-init_mask.pt'.format(i), map_location='cpu')

for key in all_masks[0].keys():
    result = []
    for i in range(8):
        result.append((all_masks[i][key]==all_masks[1][key]).float().mean().item())
    print(key, result)


all_masks = {}
for i in range(8):
    all_masks[i] = torch.load('{}-init_mask_syn.pt'.format(i), map_location='cpu')

for key in all_masks[0].keys():
    result = []
    for i in range(8):
        result.append((all_masks[i][key]==all_masks[1][key]).float().mean().item())
    print(key, result)