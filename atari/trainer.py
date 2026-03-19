import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto


class Trainer:
    def __init__(self, args, agent, optimizer, writer):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self._debug_step = 0

    def train(self, numpy_rng, global_step, b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values):
        b_index = np.arange(self.args.batch_size)

        for epoch in range(self.args.update_epochs):
            numpy_rng.shuffle(b_index)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, new_entropy, new_value = self.agent.get_action_and_value(
                    b_obs[mb_index], b_actions[mb_index]
                )

                # Probability ratio
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                if self.args.advantage_normalization:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                policy_loss = self.compute_policy_loss(ratios, mb_advantages)

                # Value loss
                value_loss = self.compute_value_loss(new_value, b_returns[mb_index], b_values[mb_index])

                # Policy entropy
                entropy_loss = new_entropy.mean()

                # Total loss
                loss = policy_loss + value_loss * self.args.c_1 - entropy_loss * self.args.c_2

                # Save the data during the training process
                self.writer.add_scalar('charts/ratio_deviation', torch.abs(ratios - 1).mean(), global_step)
                self.writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
                self.writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
                self.writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

                # Update network parameters
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

    def compute_value_loss(self, new_value, mb_returns, mb_values):
        """
        Compute value loss
        """
        new_value = new_value.view(-1)
        if self.args.clip_value_loss:
            value_loss_un_clipped = (new_value - mb_returns) ** 2
            value_clipped = mb_values + torch.clamp(new_value - mb_values, -self.args.epsilon, self.args.epsilon)
            value_loss_clipped = (value_clipped - mb_returns) ** 2
            value_loss_max = torch.max(value_loss_un_clipped, value_loss_clipped)
            value_loss = 0.5 * value_loss_max.mean()
        else:
            value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

        return value_loss

    def compute_policy_loss(self, ratios, mb_advantages):
        """
        Compute the policy loss
        """
        if self.args.algo == 'ppo':
            return self._ppo_loss(ratios, mb_advantages)

        if self.args.algo == 'spo':
            return self._spo_loss(ratios, mb_advantages)

        if self.args.algo == 'opo':
            return self._opo_loss(ratios, mb_advantages)

        if self.args.algo == 'opspo':
            return self._opspo_loss(ratios, mb_advantages)

        raise ValueError(f'unknown algo: {self.args.algo}')

    def _ppo_loss(self, ratios, mb_advantages):
        policy_loss_1 = mb_advantages * ratios
        policy_loss_2 = mb_advantages * torch.clamp(
            ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
        )
        return -torch.min(policy_loss_1, policy_loss_2).mean()

    def _spo_loss(self, ratios, mb_advantages):
        return -(
                mb_advantages * ratios -
                torch.abs(mb_advantages) * torch.pow(ratios - 1, 2) / (2 * self.args.epsilon)
        ).mean()

    def _opo_loss(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res, kappa_plus = self._fit_tail(r_safe, idx_plus, group_name='positive')
        minus_res, kappa_minus = self._fit_tail(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPO: group too small for tail fit; falling back to PPO.')
            return self._ppo_loss(r, mb_advantages)

        if max(kappa_plus, kappa_minus) > self.args.kappa_0:
            # Tail is too heavy; fallback to untruncated ratios.
            return -(r * mb_advantages).mean()

        r_tilde = r.clone()

        # Positive-advantage truncation (upper tail).
        plus_sorted, plus_order, q_plus, plus_caps = plus_res
        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        r_tilde[plus_idx] = torch.minimum(r[plus_idx], plus_caps_t)

        # Negative-advantage truncation (lower tail via flipped ratios).
        minus_sorted, minus_order, q_minus, minus_caps = minus_res
        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        r_tilde[minus_idx] = torch.maximum(r[minus_idx], minus_floor_t)

        return -(r_tilde * mb_advantages).mean()

    def _opspo_loss_advsign(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPSPO: group too small for tail caps; falling back to SPO.')
            return self._spo_loss(r, mb_advantages)

        eps = torch.full_like(r, float(self.args.epsilon))

        # Positive-advantage epsilon (upper tail).
        plus_sorted, plus_order, q_plus, plus_caps = plus_res
        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.clamp(plus_caps_t - 1.0, min=1e-8)

        # Negative-advantage epsilon (lower tail via flipped ratios).
        minus_sorted, minus_order, q_minus, minus_caps = minus_res
        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.clamp(1.0 - minus_floor_t, min=1e-8)

        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        if self.args.opspo_tail_only_penalty:
            tail_mask = torch.zeros_like(pos_mask)
            tail_mask[plus_idx] = True
            tail_mask[minus_idx] = True
            pos_tail = pos_mask & tail_mask
            neg_tail = neg_mask & tail_mask
            penalty[pos_tail] = torch.pow(r[pos_tail], 2) / (2 * eps[pos_tail])
            penalty[neg_tail] = torch.pow(1.0 / r[neg_tail], 2) / (2 * eps[neg_tail])
        else:
            penalty[pos_mask] = torch.pow(r[pos_mask], 2) / (2 * eps[pos_mask])
            penalty[neg_mask] = torch.pow(1.0 / r[neg_mask], 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _opspo_loss(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        r_flat = r.reshape(-1)
        r_safe_flat = r_safe.reshape(-1)
        if getattr(self.args, 'verbose', False) and self._debug_step % 100 == 0:
            print(f'OPSPO debug: r_safe_flat size = {r_safe_flat.numel()}')
        self._debug_step += 1
        idx_all = torch.arange(r_flat.numel(), device=device)

        all_res = self._tail_caps_fixed(r_safe_flat, idx_all, group_name='all')
        if all_res is None:
            warnings.warn('OPSPO: group too small for tail caps; falling back to SPO.')
            return self._spo_loss(r, mb_advantages)

        eps = torch.full_like(r_flat, float(self.args.epsilon))
        _, all_order, q_all, all_caps = all_res
        all_tail = all_order[q_all:]
        tail_idx = idx_all[all_tail]
        all_caps_t = torch.tensor(all_caps, device=device, dtype=dtype)
        eps[tail_idx] = torch.clamp(all_caps_t - 1.0, min=1e-8)
        eps = eps.reshape_as(r)

        return -(
                mb_advantages * r -
                torch.abs(mb_advantages) * torch.pow(r - 1, 2) / (2 * eps)
        ).mean()

    def _fit_tail(self, values, idx, group_name):
        """
        Fit a GPD to the upper tail of values[idx] and return tail caps.
        This isolates the scipy dependency so alternative fits can be swapped later.
        """
        if idx.numel() < 2:
            warnings.warn(f'OPO: {group_name} group too small for tail fit (n={idx.numel()}).')
            return None, None

        vals = values[idx]
        vals_sorted, order = vals.sort()
        s = vals_sorted.numel()
        m = int(min(0.2 * s, 3 * np.sqrt(s)))
        if m < 1 or s - m < 1:
            warnings.warn(f'OPO: {group_name} tail size too small (n={s}, m={m}).')
            return None, None

        q = s - m
        threshold = vals_sorted[q - 1].item()
        tail = vals_sorted[q:].detach().cpu().numpy()
        excess = tail - threshold
        if excess.size < 2:
            warnings.warn(f'OPO: {group_name} tail fit needs >=2 samples.')
            return None, None

        # Fit GPD to exceedances above threshold.
        shape, _, scale = genpareto.fit(excess, floc=0)
        kappa_hat = shape

        max_tail = vals_sorted[-1].item()
        p = (np.arange(1, m + 1) - 0.5) / m
        q_excess = genpareto.ppf(p, shape, loc=0, scale=scale)
        q_excess = np.nan_to_num(
            q_excess,
            nan=max_tail - threshold,
            posinf=max_tail - threshold,
            neginf=0.0,
        )
        caps = np.minimum(threshold + q_excess, max_tail)

        return (vals_sorted, order, q, caps), kappa_hat

    def _tail_caps_fixed(self, values, idx, group_name):
        """
        Compute tail caps using fixed GPD shape/scale and order statistics.
        """
        if idx.numel() < 2:
            warnings.warn(f'OPSPO: {group_name} group too small for tail caps (n={idx.numel()}).')
            return None

        vals = values[idx]
        vals_sorted, order = vals.sort()
        s = vals_sorted.numel()
        m = int(min(0.2 * s, 3 * np.sqrt(s)))
        if m < 1 or s - m < 1:
            warnings.warn(f'OPSPO: {group_name} tail size too small (n={s}, m={m}).')
            return None

        q = s - m
        threshold = vals_sorted[q - 1].item()
        max_tail = vals_sorted[-1].item()

        shape = float(self.args.gpd_shape)
        scale = float(self.args.gpd_scale)
        p = (np.arange(1, m + 1) - 0.5) / m
        q_excess = genpareto.ppf(p, shape, loc=0, scale=scale)
        q_excess = np.nan_to_num(
            q_excess,
            nan=max_tail - threshold,
            posinf=max_tail - threshold,
            neginf=0.0,
        )
        caps = np.minimum(threshold + q_excess, max_tail)

        return (vals_sorted, order, q, caps)
