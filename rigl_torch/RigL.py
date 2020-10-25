""" implementation of https://arxiv.org/abs/1911.11134 """

import numpy as np
import torch

from rigl_torch.util import get_W


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]
        self.dense_grad = grad.clone()
        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class RigLScheduler:

    def __init__(self, model, optimizer, dense_allocation=1, T_end=None, sparsity_distribution='uniform', ignore_linear_layers=True, is_already_sparsified=False, delta=100, alpha=0.3, static_topo=False):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception('Dense allocation must be on the interval (0, 1]. Got: %f' % dense_allocation)

        self.model = model
        self.optimizer = optimizer
        self.sparsity_distribution = sparsity_distribution
        self.static_topo = static_topo
        self.backward_masks = None

        assert self.sparsity_distribution in ('uniform', )

        self.W = get_W(model, ignore_linear_layers=ignore_linear_layers)
        # when using uniform sparsity, the first layer is always 100% dense
        self.W = self.W[1:]

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)

        # define sparsity allocation
        layers = len(self.W)
        self.S = [1-dense_allocation] * layers # uniform sparsity
        self.N = [torch.numel(w) for w in self.W]

        # randomly sparsify model according to S
        if is_already_sparsified:
            raise Exception('TODO')

            # get zeros masks
            self.backward_masks = []
            for l, w in enumerate(self.W):
                flat_w = w.view(-1)
                mask = torch.abs(flat_w) == 0
                indices = torch.arange(len(flat_w))
                mask = indices[mask]
                self.backward_masks.append(mask)
        else:
            self.random_sparsify()

        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])

        # scheduler keeps a log of how many times it's called. this is how it does its scheduling
        self.step = 0
        self.rigl_steps = 0

        # define the actual schedule
        self.delta_T = delta
        self.alpha = alpha
        self.T_end = T_end

    @torch.no_grad()
    def random_sparsify(self):
        self.backward_masks = []
        for l, w in enumerate(self.W):
            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)
            perm = perm[:s]
            flat_mask = torch.ones(n, dtype=torch.bool, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)
            w *= mask
            self.backward_masks.append(mask)


    def __call__(self):
        self.step += 1
        if (self.step % self.delta_T) == 0 and self.step < self.T_end: # check schedule
            self._rigl_step()
            self.rigl_steps += 1
            return False
        return True


    def __str__(self):
        s = 'RigLScheduler(\n'
        s += 'layers=%i,\n' % len(self.N)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = '['
        S_str = '['
        sparsity_percentages = []
        total_params = 0
        total_nonzero = 0

        for N, S, mask, W in zip(self.N, self.S, self.backward_masks, self.W):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += ('%i/%i, ' % (N-actual_S, N))
            sp_p = float(N-actual_S) / float(N) * 100
            S_str += '%.2f%%, ' % sp_p

            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N-actual_S
        N_str = N_str[:-2] + ']'
        S_str = S_str[:-2] + ']'
        
        s += 'nonzero_params=' + N_str + ',\n'
        s += 'nonzero_percentages=' + S_str + ',\n'
        s += 'total_nonzero_params=' + ('%i/%i (%.2f%%)' % (total_nonzero, total_params, float(total_nonzero)/float(total_params)*100)) + ',\n'
        s += 'step=' + str(self.step) + ',\n'
        s += 'num_rigl_steps=' + str(self.rigl_steps) + ',\n'

        return s + ')'


    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))


    @torch.no_grad()
    def reset_momentum(self):
        for w, mask in zip(self.W, self.backward_masks):
            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask in zip(self.W, self.backward_masks):
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask in zip(self.W, self.backward_masks):
            w.grad *= mask


    @torch.no_grad()
    def _rigl_step(self):
        if self.static_topo:
            return 

        drop_fraction = self.cosine_annealing()

        for l, w in enumerate(self.W):
            current_mask = self.backward_masks[l]

            # calculate raw scores
            score_drop = torch.abs(w)
            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)

            # calculate drop/grow quantities
            n_total = self.N[l]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune

            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)

            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_prune,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)
            
            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            # update new weights to be initialized as zeros and update the weight tensors
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            # update the mask
            current_mask.data = mask_combined

            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients() 
