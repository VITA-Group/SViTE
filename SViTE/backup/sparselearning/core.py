from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
import numpy as np
import math
import pdb

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--final_density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--snip', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization')
    parser.add_argument('--reset', action='store_true', help='Fix topology during training. Default: True.')


def prRed(skk): print("\033[91m{}\033[00m".format(skk))


def prGreen(skk): print("\033[92m{}\033[00m".format(skk))


def prYellow(skk): print("\033[93m{}\033[00m".format(skk))


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']


class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate


def parameters_distribution(model):

    emb_all = 0
    mlp_all = 0
    att_mlp_all = 0
    att_qkv_all = 0
    others = 0
    for name, tensor in model.named_parameters():
        if 'embed.proj' in name:
            emb_all += tensor.numel()
        elif 'attn.proj' in name:
            att_mlp_all += tensor.numel()
        elif 'attn.qkv' in name:
            att_qkv_all += tensor.numel()
        elif 'mlp' in name:
            mlp_all += tensor.numel()
        else:
            others += tensor.numel()
    total = emb_all + att_mlp_all + att_qkv_all + mlp_all + others
    print("all:{}".format(total))
    print("embeding:{} /{:.2f}".format(emb_all, emb_all/total))
    print("attn mlp:{} /{:.2f}".format(att_mlp_all, att_mlp_all/total))
    print("attn qkv:{} /{:.2f}".format(att_qkv_all, att_qkv_all/total))
    print("mlp all :{} /{:.2f}".format(mlp_all, mlp_all/total))
    print("others  :{} /{:.2f}".format(others, others/total))


class Masking(object):
    def __init__(self,
            args=None,
            optimizer, 

            death_rate=0.3, 
            growth_death_ratio=1.0, 
            death_rate_decay=None, 
            death_mode='magnitude',

            growth_mode='momentum', 
            redistribution_mode='momentum', 

            spe_initial=None, 
            train_loader=None, 
            device_ids=0):

        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))

        self.args = args
        self.loader = train_loader
        self.device = torch.device("cuda:{}".format(device_ids))
        self.growth_mode = growth_mode # gradient
        self.death_mode = death_mode   # magnitude
        self.redistribution_mode = redistribution_mode # momentum
        self.death_rate_decay = death_rate_decay
        self.spe_initial = spe_initial # initial masks made by SNIP
        self.snip_masks = None # masks made by SNIP during training
        self.nonzeros_index = None
        self.masks = {}
        self.atten_masks = {}
        self.other_masks = {}

        self.newly_masks = {}
        self.survival = {}
        self.pruned_number = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        # stats
        self.name2zeros = {}
        self.name2nonzeros = {}

        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0

    def structure_init(self, mode='ER', erk_power_scale=1.0):
        
        #############################
        # for mlp
        #############################
        if mode == 'uniform':
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur in self.other_masks:
                        self.other_masks[name_cur][:] = (torch.rand(weight.shape) < self.args.other_density).float().data.cuda()    
                    else: continue

        elif mode == 'fixed_ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            for name, weight in self.other_masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.other_masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - self.args.other_density) # 0.95
                    n_ones = n_param * self.args.other_density        # 0.05

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True
            
            density_dict = {}
            total_nonzero = 0.0
            for name, mask in self.other_masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.other_masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        elif mode == 'ER':
            print('initialize by SET')
            # initialization used in sparse evolutionary training
            total_params = 0
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.other_masks: continue
                    total_params += weight.numel()

            target_params = total_params *self.args.other_density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                index = 0
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.other_masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            index = 0
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.other_masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.other_masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data.cuda()

        #############################
        # for attention
        #############################
        index = 0
        atten_list = []
        for module in self.modules:
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur in self.atten_masks:
                    atten_list.append((name_cur, weight.shape))
                    atten_dim = weight.shape[-1]
                    atten_dim0 = weight.shape[0]
                else: continue

        self.num_atten = len(atten_list) * self.args.atten_head # 12 * 3 = 36
        self.nonzeros_index = random.sample([i for i in range(self.num_atten)], int(self.num_atten * self.args.atten_density))

        muti_head_dim = int(atten_dim / self.args.atten_head)
        self.atten_mask_shape = (atten_dim0, muti_head_dim)
        self.atten_key_index = [k for k in self.atten_masks.keys()]
        print("-" * 100)

        for nonzero_idx in self.nonzeros_index:
            
            key = self.atten_key_index[int(nonzero_idx / self.args.atten_head)]
            left = (nonzero_idx % self.args.atten_head) * muti_head_dim
            right = ((nonzero_idx + 1) % self.args.atten_head) * muti_head_dim
            if right == 0: right = muti_head_dim * self.args.atten_head
            self.atten_masks[key][:, left : right] = torch.ones(self.atten_mask_shape).float().data.cuda()

            print('{} | {}/{} | shape:{}'.format(key, 
                self.atten_masks[key].sum().int().item(), 
                self.atten_masks[key].numel(), self.atten_mask_shape))
        
        self.apply_mask(pruning_type="structure")
        self.fired_masks = copy.deepcopy(self.other_masks) # used for over-paremeters
        self.init_death_rate(self.death_rate, pruning_type="structure")
        self.print_structure_mask()

    def init(self, mode='ER', density=0.05, erk_power_scale=1.0, mask_file=None):
        self.sparsity = density
        if mode == 'uniform':
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data.cuda()

        if mode == 'custom':
            assert mask_file
            print('* custom init mask')
            custom_mask = torch.load(mask_file, map_location=self.device)
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)

                    if 'module' in name:
                        mask_name = name[len('module.'):] + '_mask'
                    else:
                        mask_name = name + '_mask'

                    index += 1
                    if name_cur not in self.masks: continue
                    print(name, mask_name in custom_mask.keys())
                    self.masks[name_cur][:] = custom_mask[mask_name]

        elif mode == 'fixed_ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density) # 0.95
                    n_ones = n_param * density        # 0.05

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True
            
            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        elif mode == 'ER':
            print('initialize by SET')
            # initialization used in sparse evolutionary training
            total_params = 0
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    total_params += weight.numel()

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                index = 0
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            index = 0
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data.cuda()


        self.apply_mask()
        self.fired_masks = copy.deepcopy(self.masks) # used for over-paremeters
        self.init_death_rate(self.death_rate)

        # self.print_nonzero_counts()

        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()
        
        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))

    def init_death_rate(self, death_rate, pruning_type="unstructure"):
        
        if pruning_type == "unstructure":
            for name in self.masks:
                self.name2death_rate[name] = death_rate
        elif pruning_type == "structure":
            for name in self.other_masks:
                self.name2death_rate[name] = death_rate
            for name in self.atten_masks:
                self.name2death_rate[name] = death_rate
        else: assert False

    def at_end_of_epoch(self, pruning_type="unstructure", indicator_list=None):
        if pruning_type == "unstructure":
            self.truncate_weights()
            _, _ = self.fired_masks_update()
            self.print_nonzero_counts()
        elif pruning_type == "structure":
            self.truncate_weights(pruning_type, indicator_list)
            _, _ = self.fired_masks_update(pruning_type="structure")
        else:
            assert False

    def step(self, pruning_type="unstructure"):
        
        # self.optimizer.step()
        self.apply_mask(pruning_type=pruning_type)
        self.death_rate_decay.step()
        for name in self.masks:
            if self.args.decay_schedule == 'cosine':
                self.name2death_rate[name] = self.death_rate_decay.get_dr(self.name2death_rate[name])
            elif self.args.decay_schedule == 'constant':
                self.name2death_rate[name] = self.args.death_rate
            self.death_rate = self.name2death_rate[name]
        self.steps += 1

    def add_module(self, module, density, sparse_init='ER', pruning_type="unstructure", mask_path=None):

        if pruning_type == 'unstructure':
        
            self.modules.append(module)
            index = 0
            for name, tensor in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if len(tensor.size()) ==4 or len(tensor.size()) ==2:
                    self.names.append(name_cur)
                    self.masks[name_cur] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            
            print('Removing biases...')
            self.remove_weight_partial_name('bias')
            # print('Removing 2D batch norms...')
            # self.remove_type(nn.BatchNorm2d)
            # print('Removing 1D batch norms...')
            # self.remove_type(nn.BatchNorm1d)
            self.init(mode=sparse_init, density=density, mask_file=mask_path)
            # self.approximnate_isometry()
        elif pruning_type == 'structure':
            
            parameters_distribution(module)
            self.modules.append(module)
            index = 0
            for name, tensor in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if len(tensor.size()) ==4 or len(tensor.size()) ==2:
                    self.names.append(name_cur)
                    if 'attn.qkv' in name_cur:
                        self.atten_masks[name_cur] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                    elif 'attn.proj' not in name_cur: # no pruning attention mlp
                        self.other_masks[name_cur] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            
            print('Removing biases...')
            self.remove_weight_partial_name('bias')
            self.structure_init(mode=sparse_init)
        else: assert False

    def remove_weight(self, name, index):

        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, 
                                                                      self.masks[name].shape,
                                                                      self.masks[name].numel()))

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        index = 0
        for module in self.modules:
            for name, module in module.named_modules():
                print(name)
                if isinstance(module, nn_type):
                    self.remove_weight(name, index)
                index += 1

    def apply_mask(self, pruning_type="unstructure"):

        if pruning_type=="unstructure":
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name+'_'+str(index)
                    index += 1
                    if name_cur in self.masks:
                        weight.data = weight.data*self.masks[name_cur]
                        if 'momentum_buffer' in self.optimizer.state[weight]:
                            self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer']*self.masks[name_cur]

        elif pruning_type=="structure":
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name+'_'+str(index)
                    index += 1
                    if name_cur in self.other_masks:
                        weight.data = weight.data * self.other_masks[name_cur]
                        if 'momentum_buffer' in self.optimizer.state[weight]:
                            self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer']*self.other_masks[name_cur]
                    elif name_cur in self.atten_masks:
                        weight.data = weight.data * self.atten_masks[name_cur]
                        if 'momentum_buffer' in self.optimizer.state[weight]:
                            self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer']*self.atten_masks[name_cur]
                    else: continue
        else:
            assert False

    def truncate_weights(self, pruning_type="unstructure", indicator_list=None):

        if pruning_type=="unstructure":
            self.gather_statistics() # count each of module's zeros and non-zeros
            #prune
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    mask = self.masks[name_cur]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name_cur)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name_cur)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name_cur)

                    self.pruned_number[name_cur] = int(self.name2nonzeros[name_cur] - new_mask.sum().item()) # record pruning numbers
                    self.masks[name_cur][:] = new_mask  # update new mask
            #grow
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name +'_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    new_mask = self.masks[name_cur].data.byte()

                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    elif self.growth_mode == 'gradient':
                        # implementation for Rigging Ticket
                        new_mask = self.gradient_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    # exchanging masks
                    self.masks.pop(name_cur)
                    self.masks[name_cur] = new_mask.float()

            self.apply_mask()

        elif pruning_type=="structure":

            self.gather_statistics(pruning_type="structure") # count each of module's zeros and non-zeros

            ##########################
            # FOR MLP
            ##########################
            #prune
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.other_masks: continue
                    mask = self.other_masks[name_cur]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name_cur)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name_cur)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name_cur)

                    self.pruned_number[name_cur] = int(self.name2nonzeros[name_cur] - new_mask.sum().item()) # record pruning numbers
                    self.other_masks[name_cur][:] = new_mask  # update new mask
            #grow
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name +'_' + str(index)
                    index += 1
                    if name_cur not in self.other_masks: continue
                    new_mask = self.other_masks[name_cur].data.byte()

                    if self.growth_mode == 'random':
                        new_mask = self.random_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    elif self.growth_mode == 'momentum':
                        new_mask = self.momentum_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    elif self.growth_mode == 'gradient':
                        # implementation for Rigging Ticket
                        new_mask = self.gradient_growth(name_cur, new_mask, self.pruned_number[name_cur], weight)

                    # exchanging masks
                    self.other_masks.pop(name_cur)
                    self.other_masks[name_cur] = new_mask.float()

            ##########################
            # FOR attention
            ##########################
            # prune
            prRed("-" * 100)
            prRed("begin death attention (l1 indicator)")
            prRed("-" * 100)
            indicator_list = torch.tensor(indicator_list).reshape(-1)
            _, sort_index = torch.sort(indicator_list[torch.tensor(self.nonzeros_index)])
            pruned_num = int(self.args.death_rate * len(sort_index))
            pruned_index = []
            for i in sort_index[:pruned_num]:
                pruned_index.append(self.nonzeros_index[i])
            
            # update nonzero_index
            self.nonzeros_index = [i for i in self.nonzeros_index if i not in pruned_index]
            muti_head_dim = self.atten_mask_shape[-1]

            for zero_idx in pruned_index:
                
                key = self.atten_key_index[int(zero_idx / self.args.atten_head)]
                left = (zero_idx % self.args.atten_head) * muti_head_dim
                right = ((zero_idx + 1) % self.args.atten_head) * muti_head_dim
                if right == 0: right = muti_head_dim * self.args.atten_head
                self.atten_masks[key][:, left : right] = torch.zeros(self.atten_mask_shape).float().data.cuda()

            prGreen("-" * 100)
            prGreen("begin grow attention (random)")
            prGreen("-" * 100)
            # grow random
            zero_idx = [i for i in range(self.num_atten) if i not in self.nonzeros_index]
            grow_index = random.sample(zero_idx, pruned_num)
            for idx in grow_index:
            
                key = self.atten_key_index[int(idx / self.args.atten_head)]
                left = (idx % self.args.atten_head) * muti_head_dim
                right = ((idx + 1) % self.args.atten_head) * muti_head_dim
                if right == 0: right = muti_head_dim * self.args.atten_head
                self.atten_masks[key][:, left : right] = torch.ones(self.atten_mask_shape).float().data.cuda()
            
            self.apply_mask(pruning_type="structure")
            self.print_structure_mask()
            
        else: assert False        

    def gather_statistics(self, pruning_type="unstructure"):
        self.name2nonzeros = {}
        self.name2zeros = {}

        if pruning_type=="unstructure":
            index = 0
            for module in self.modules:
                for name, tensor in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    mask = self.masks[name_cur]
                    self.name2nonzeros[name_cur] = mask.sum().item()
                    self.name2zeros[name_cur] = mask.numel() - self.name2nonzeros[name_cur]

        elif pruning_type=="structure":

            index = 0
            for module in self.modules:
                for name, tensor in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur in self.other_masks:
                        mask = self.other_masks[name_cur]
                        self.name2nonzeros[name_cur] = mask.sum().item()
                        self.name2zeros[name_cur] = mask.numel() - self.name2nonzeros[name_cur]
                    elif name_cur in self.atten_masks:
                        mask = self.atten_masks[name_cur]
                        self.name2nonzeros[name_cur] = mask.sum().item()
                        self.name2zeros[name_cur] = mask.numel() - self.name2nonzeros[name_cur]
        else:
            assert False


    '''
                DEATH
    '''
    def CS_death(self,  mask,  snip_mask):
        # calculate scores for all weights
        # note that the gradients are from the last iteration, which are not very accurate
        # but in another perspective, we can understand the weights are from the next iterations, the differences are not very large.
        '''
        grad = self.get_gradient_for_weights(weight)
        scores = torch.abs(grad * weight * (mask == 0).float())
        norm_factor = torch.sum(scores)
        scores.div_(norm_factor)
        x, idx = torch.sort(scores.data.view(-1))

        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        if num_remove == 0.0: return weight.data != 0.0

        mask.data.view(-1)[idx[:k]] = 0.0
        '''

        assert (snip_mask.shape == mask.shape)

        return snip_mask

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def magnitude_death(self, mask, weight, name):

        if mask.sum().item() == mask.numel():
            return mask

        death_rate = self.name2death_rate[name]

        num_remove = math.ceil(death_rate*self.name2nonzeros[name]) # pruning nonzeros
        if num_remove == 0.0: return weight.data != 0.0
        #num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        n = idx.shape[0]

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k-1].item()

        return (torch.abs(weight.data) > threshold)

    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                GROWTH
    '''

    def random_growth(self, name, new_mask, total_regrowth, weight):
        n = (new_mask==0).sum().item()
        if n == 0: return new_mask
        expeced_growth_probability = (total_regrowth/n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        # for pytorch1.5.1, use return new_mask.bool() | new_weights
        return new_mask.byte() | new_weights

    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if total_regrowth == 0:
            return new_mask
        grad = self.get_gradient_for_weights(weight)
        grad = grad*(new_mask==0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def mix_growth(self, name, new_mask, total_regrowth, weight):
        gradient_grow = int(total_regrowth * self.args.mix)
        random_grow = total_regrowth - gradient_grow
        grad = self.get_gradient_for_weights(weight)
        grad = grad * (new_mask == 0).float()

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:gradient_grow]] = 1.0

        n = (new_mask == 0).sum().item()
        expeced_growth_probability = (random_grow / n)
        new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
        new_mask = new_mask.byte() | new_weights

        return new_mask, grad

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        index = 0
        for module in self.modules:
            for name, tensor in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                mask = self.masks[name_cur]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name_cur, self.name2nonzeros[name_cur], num_nonzeros, num_nonzeros/float(mask.numel()))
                print(val)
        print('Death rate: {0}\n'.format(self.death_rate))

    def print_structure_mask(self):
        
        prYellow("=" * 100)
        prYellow("Mask INFO")
        prYellow("=" * 100)

        mlp_total_size = 0
        att_total_size = 0
        mlp_sparse_size = 0
        att_sparse_size = 0

        for name, weight in self.other_masks.items():
            mlp_total_size  += weight.numel()
            mlp_sparse_size += (weight != 0).sum().int().item()

        prYellow("-" * 100)
        for name, weight in self.atten_masks.items():
            print('{} | {}/{} | shape:{}'.format(name, (weight != 0).sum().int().item(), weight.numel(), weight.shape))
            att_total_size  += weight.numel()
            att_sparse_size += (weight != 0).sum().int().item()

        prYellow("-" * 100)
        prYellow('Total parameters under sparsity level of mlp [{}/{:.4f}] att [{}/{:.4f}]'
                .format(self.args.other_density, 
                        mlp_sparse_size / mlp_total_size, 
                        self.args.atten_density, 
                        att_sparse_size / att_total_size))
        prYellow("-" * 100)

    def fired_masks_update(self, pruning_type='unstructure'):

        if pruning_type == 'unstructure':
            ntotal_fired_weights = 0.0
            ntotal_weights = 0.0
            layer_fired_weights = {}
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    self.fired_masks[name_cur] = self.masks[name_cur].data.byte() | self.fired_masks[name_cur].data.byte() # count fired weights in certain layer
                    ntotal_fired_weights += float(self.fired_masks[name_cur].sum().item())# count total fired weights in certain layer
                    ntotal_weights += float(self.fired_masks[name_cur].numel()) # count total weight
                    layer_fired_weights[name_cur] = float(self.fired_masks[name_cur].sum().item())/float(self.fired_masks[name_cur].numel()) # percents
                    print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name_cur])
            total_fired_weights = ntotal_fired_weights/ntotal_weights
            print('The percentage of the total fired weights is:', total_fired_weights)
            return layer_fired_weights, total_fired_weights
        else:
            ntotal_fired_weights = 0.0
            ntotal_weights = 0.0
            layer_fired_weights = {}
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.other_masks: continue
                    self.fired_masks[name_cur] = self.other_masks[name_cur].data.byte() | self.fired_masks[name_cur].data.byte() # count fired weights in certain layer
                    ntotal_fired_weights += float(self.fired_masks[name_cur].sum().item())# count total fired weights in certain layer
                    ntotal_weights += float(self.fired_masks[name_cur].numel()) # count total weight
                    layer_fired_weights[name_cur] = float(self.fired_masks[name_cur].sum().item())/float(self.fired_masks[name_cur].numel()) # percents
                    print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name_cur])
            total_fired_weights = ntotal_fired_weights/ntotal_weights
            print('The percentage of the total fired weights is:', total_fired_weights)
            return layer_fired_weights, total_fired_weights


