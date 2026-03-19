import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import genpareto


def compute_kld(mu_1, sigma_1, mu_2, sigma_2):
    return torch.log(sigma_2 / sigma_1) + ((mu_1 - mu_2) ** 2 + (sigma_1 ** 2 - sigma_2 ** 2)) / (2 * sigma_2 ** 2)


class Trainer:
    def __init__(self, args, agent, optimizer, writer):
        self.args = args
        self.agent = agent
        self.optimizer = optimizer
        self.writer = writer
        self._debug_step = 0

    def train(
            self, global_step, b_obs, b_actions, b_log_probs, b_advantages, b_returns,
            b_values, b_mean, b_std, current_update=None, total_updates=None
    ):
        b_index = np.arange(self.args.batch_size)

        if self.args.algo == 'ppo_lambda':
            lambda_n = self._current_lambda(current_update=current_update, total_updates=total_updates)
            epsilon = self._current_clip_epsilon(current_update=current_update, total_updates=total_updates)
            self.writer.add_scalar('charts/ppo_lambda_current_lambda', lambda_n, global_step)
            self.writer.add_scalar('charts/ppo_lambda_current_epsilon', epsilon, global_step)

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_index)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, new_entropy, new_value, new_mean_std = self.agent.get_action_and_value(
                    b_obs[mb_index], b_actions[mb_index]
                )

                # Probability ratio
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                # Adaptive learning rate for ppo
                new_mean = new_mean_std.loc.reshape(self.args.minibatch_size, -1)
                new_std = new_mean_std.scale.reshape(self.args.minibatch_size, -1)
                kl = compute_kld(b_mean[mb_index], b_std[mb_index], new_mean, new_std).sum(-1)
                if self.args.adaptive_learning_rate and self.args.algo == 'ppo':
                    if kl.mean() > self.args.desired_kl * 2.0:
                        self.args.learning_rate = max(1e-5, self.args.learning_rate / 1.5)
                        self.optimizer.param_groups[0]['lr'] = self.args.learning_rate
                    if kl.mean() < self.args.desired_kl / 2.0:
                        self.args.learning_rate = min(1e-2, self.args.learning_rate * 1.5)
                        self.optimizer.param_groups[0]['lr'] = self.args.learning_rate
                self.writer.add_scalar('This is for plotting/kl_divergence', kl.mean(), global_step)

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                if self.args.advantage_normalization:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                policy_loss = self.compute_policy_loss_from_log_ratio(
                    log_ratio, ratios, mb_advantages, global_step=global_step,
                    current_update=current_update, total_updates=total_updates
                )

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

    def compute_policy_loss(self, ratios, mb_advantages, global_step=None, current_update=None, total_updates=None):
        """
        Compute the policy loss
        """
        if self.args.algo == 'ppo':
            return self._ppo_loss(ratios, mb_advantages)

        if self.args.algo == 'spo':
            return self._spo_loss(ratios, mb_advantages)

        if self.args.algo == 'opo':
            return self._opo_loss(ratios, mb_advantages)

        if self.args.algo == 'opo-penalty':
            log_ratio = torch.log(ratios.clamp_min(1e-8))
            return self._opo_penalty_loss(
                log_ratio=log_ratio,
                ratios=ratios,
                mb_advantages=mb_advantages,
                global_step=global_step,
            )

        if self.args.algo == 'opspo':
            return self._opspo_loss(ratios, mb_advantages)

        if self.args.algo == 'opspo_naive':
            return self._opspo_naive_loss(ratios, mb_advantages)

        if self.args.algo == 'oppo_ranked_clip':
            return self._oppo_ranked_clip_loss(ratios, mb_advantages)

        if self.args.algo == 'opspo_fixed':
            return self._opspo_fixed_loss(ratios, mb_advantages)

        if self.args.algo == 'opspo_fixed_adv':
            return self._opspo_fixed_adv_loss(ratios, mb_advantages)

        if self.args.algo == 'opspo_fixed_anneal':
            return self._opspo_fixed_anneal_loss(
                ratios, mb_advantages, global_step=global_step,
                current_update=current_update, total_updates=total_updates
            )

        if self.args.algo == 'ppo_lambda':
            # Preserve the existing signature by reconstructing a sampled-action
            # log-ratio for callers that only pass ratios.
            log_ratio = torch.log(ratios.clamp_min(1e-8))
            return self._ppo_lambda_loss(
                log_ratio=log_ratio,
                ratios=ratios,
                mb_advantages=mb_advantages,
                current_update=current_update,
                total_updates=total_updates,
            )

        raise ValueError(f'unknown algo: {self.args.algo}')

    def compute_policy_loss_from_log_ratio(
            self, log_ratio, ratios, mb_advantages, global_step=None, current_update=None, total_updates=None
    ):
        """
        Dispatch policy loss variants that may need sampled-action log-ratios.
        Existing baselines continue to use compute_policy_loss unchanged.
        """
        if self.args.algo == 'ppo_lambda':
            return self._ppo_lambda_loss(
                log_ratio=log_ratio,
                ratios=ratios,
                mb_advantages=mb_advantages,
                current_update=current_update,
                total_updates=total_updates,
            )
        if self.args.algo == 'opo-penalty':
            return self._opo_penalty_loss(
                log_ratio=log_ratio,
                ratios=ratios,
                mb_advantages=mb_advantages,
                global_step=global_step,
            )
        return self.compute_policy_loss(
            ratios,
            mb_advantages,
            global_step=global_step,
            current_update=current_update,
            total_updates=total_updates,
        )

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

    def _opo_penalty_loss(self, log_ratio, ratios, mb_advantages, global_step=None):
        shape = float(self.args.gpd_shape)
        if shape <= 0:
            raise ValueError('opo-penalty requires args.gpd_shape > 0')

        alpha = 0.7 / shape
        lambda_coef = float(getattr(self.args, 'lambda_0', 1.0))
        log_u = float(np.log1p(self.args.epsilon))

        excess = torch.relu(log_ratio - log_u)
        penalty_exponent = torch.as_tensor(alpha, device=ratios.device, dtype=ratios.dtype) * excess
        penalty_term = torch.logsumexp(penalty_exponent, dim=0) - np.log(penalty_exponent.numel())
        objective = (ratios * mb_advantages).mean() - lambda_coef * penalty_term

        if global_step is not None:
            self.writer.add_scalar('charts/opo_penalty_alpha', alpha, global_step)
            self.writer.add_scalar('charts/opo_penalty_lambda', lambda_coef, global_step)
            self.writer.add_scalar('losses/opo_penalty_term', penalty_term.item(), global_step)

        return -objective

    def _current_clip_epsilon(self, current_update=None, total_updates=None):
        """
        PPO-lambda uses PPO clip epsilon as the paper's delta.
        This codepath already reads the current epsilon from args, so we return it
        directly instead of introducing new clip-range annealing logic here.
        """
        return self.args.epsilon

    def _current_lambda(self, current_update=None, total_updates=None):
        delta_n = self._current_clip_epsilon(current_update=current_update, total_updates=total_updates)
        delta_0 = getattr(self.args, 'initial_epsilon', self.args.epsilon)
        lambda_0 = getattr(self.args, 'lambda_0', 1.0)
        eps = 1e-8
        delta_n = max(float(delta_n), eps)
        delta_0 = max(float(delta_0), eps)
        return lambda_0 * np.log1p(delta_0) / np.log1p(delta_n)

    def _ppo_lambda_loss(self, log_ratio, ratios, mb_advantages, current_update=None, total_updates=None):
        """
        PPO-lambda sampled-action surrogate based on Eq. (18) of the paper.

        This implementation uses PPO clip epsilon as the paper's delta, keeps the
        codebase's normalized-advantage behavior, and uses lambda_0=1.0 by default.
        lambda_n follows the paper's adaptive schedule.

        Because this trainer only has sampled-action log-probs, this is a sampled-
        action approximation of the paper's surrogate. The exact state-wise
        normalization constant of the target policy is not available in this path.
        """
        epsilon = self._current_clip_epsilon(current_update=current_update, total_updates=total_updates)
        lambda_n = self._current_lambda(current_update=current_update, total_updates=total_updates)
        lambda_t = torch.as_tensor(lambda_n, device=ratios.device, dtype=ratios.dtype)

        target_log_ratio = mb_advantages / lambda_t
        upper = 1.0 + epsilon
        lower = 1.0 - epsilon

        weight = torch.where(
            (mb_advantages > 0) & (ratios > upper),
            torch.full_like(ratios, upper),
            torch.where(
                (mb_advantages < 0) & (ratios < lower),
                torch.full_like(ratios, lower),
                ratios,
            ),
        )

        loss = lambda_t * weight * (log_ratio - target_log_ratio)
        return loss.mean()

    def _opo_loss_dep(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype
        base_eps = float(self.args.epsilon)
        base_eps = float(self.args.epsilon)

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res, kappa_plus = self._fit_tail(r_safe, idx_plus, group_name='positive')
        minus_res, kappa_minus = self._fit_tail(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPO: group too small for tail fit; falling back to PPO.')
            return self._ppo_loss(r, mb_advantages)

        if max(kappa_plus, kappa_minus) < self.args.kappa_0:
            # Tail good tail
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

    def _opspo_loss(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed_threshold_epsilon(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed_threshold_epsilon(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            raise ValueError
            return self._spo_loss(r, mb_advantages)

        eps = torch.full_like(r, float(self.args.epsilon))

        # Positive-advantage epsilon (upper tail).
        plus_sorted, plus_order, q_plus, plus_caps = plus_res
        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t-1)
        
        #eps[plus_idx] = torch.clamp(plus_caps_t, min=self.args.epsilon)
        # Negative-advantage epsilon (lower tail via flipped ratios).
        minus_sorted, minus_order, q_minus, minus_caps = minus_res
        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1-minus_floor_t)
        #eps = np.maximum(eps, self.args.epsilon)
        eps = torch.clamp(eps, min=self.args.epsilon)
        #eps[minus_idx] = torch.clamp(minus_floor_t, min=self.args.epsilon)
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r) 
        #if self.args.opspo_tail_only_penalty:
        #    tail_mask = torch.zeros_like(pos_mask)
        #    tail_mask[plus_idx] = True
        #    tail_mask[minus_idx] = True
        #    pos_tail = pos_mask & tail_mask
        #    neg_tail = neg_mask & tail_mask
        #    penalty[pos_tail] = torch.pow(r[pos_tail]-1, 2) / (2 * eps[pos_tail])
        #    penalty[neg_tail] = torch.pow(r[neg_tail]-1, 2) / (2 * eps[neg_tail])
            #penalty[neg_tail] = torch.pow(r[neg_tail]-1, 2) / (2 * eps[neg_tail])
    #else:
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _opspo_naive_loss(self, ratios, mb_advantages):
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed(1.0 / r_safe, idx_minus, group_name='negative')

        #if plus_res is None or minus_res is None:
        #    raise ValueError
        #    return self._spo_loss(r, mb_advantages)

        eps = torch.full_like(r, float(self.args.epsilon))

        # Positive-advantage epsilon (upper tail).
        _, plus_order, q_plus, plus_caps = plus_res
        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t - 1)

        # Negative-advantage epsilon (lower tail via flipped ratios).
        _, minus_order, q_minus, minus_caps = minus_res
        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1 - minus_floor_t)

        eps = torch.clamp(eps, min=self.args.epsilon)
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _opspo_loss_adv(self, ratios, mb_advantages):
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
        eps[tail_idx] = torch.clamp(all_caps_t, min=1e-8)
        eps = eps.reshape_as(r)

        return -(
                mb_advantages * r -
                torch.abs(mb_advantages) * torch.pow(r - 1, 2) / (2 * eps)
        ).mean()

    def _oppo_ranked_clip_loss_dep(self, ratios, mb_advantages):
        """
        Deprecated ranked-clipping variant that only clips the positive group
        on the upper side and the negative group on the lower side.
        """
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed_threshold_epsilon(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed_threshold_epsilon(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPPO_RANKED_CLIP: group too small for ranked clipping; falling back to PPO.')
            return self._ppo_loss(r, mb_advantages)

        _, plus_order, q_plus, plus_caps, _ = plus_res
        _, minus_order, q_minus, minus_caps, _ = minus_res

        upper = torch.full_like(r, 1.0 + base_eps)
        lower = torch.full_like(r, 1.0 - base_eps)

        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        upper[plus_idx] = plus_caps_t

        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        lower[minus_idx] = minus_floor_t

        clipped_r = r.clone()
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        clipped_r[pos_mask] = torch.minimum(r[pos_mask], upper[pos_mask])
        clipped_r[neg_mask] = torch.maximum(r[neg_mask], lower[neg_mask])

        policy_loss_1 = mb_advantages * r
        policy_loss_2 = mb_advantages * clipped_r
        return -torch.min(policy_loss_1, policy_loss_2).mean()

    def _oppo_ranked_clip_loss(self, ratios, mb_advantages):
        """
        PPO-style clipped surrogate with per-sample clip widths determined by
        the current ratio rank within each advantage-sign group.

        Positive-advantage samples are clipped to [1 - eps, 1 + upper].
        Negative-advantage samples are clipped to [1 - lower, 1 + eps].
        Both upper and lower widths are floored at epsilon.
        """
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype
        base_eps = float(self.args.epsilon)

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed_threshold_epsilon(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed_threshold_epsilon(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPPO_RANKED_CLIP: group too small for ranked clipping; falling back to PPO.')
            return self._ppo_loss(r, mb_advantages)

        _, plus_order, q_plus, plus_caps, _ = plus_res
        _, minus_order, q_minus, minus_caps, _ = minus_res

        pos_upper_bound = torch.full_like(r, 1.0 + base_eps)
        neg_lower_bound = torch.full_like(r, 1.0 - base_eps)

        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        pos_upper_bound[plus_idx] = torch.maximum(
            plus_caps_t,
            torch.full_like(plus_caps_t, 1.0 + base_eps),
        )

        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_caps_t = torch.tensor(minus_caps, device=device, dtype=dtype)
        neg_lower_bound[minus_idx] = torch.minimum(
            1.0 / minus_caps_t,
            torch.full_like(minus_caps_t, 1.0 - base_eps),
        )

        clipped_r = r.clone()
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        clipped_r[pos_mask] = torch.minimum(
            torch.maximum(r[pos_mask], torch.full_like(r[pos_mask], 1.0 - base_eps)),
            pos_upper_bound[pos_mask],
        )
        clipped_r[neg_mask] = torch.minimum(
            torch.maximum(r[neg_mask], neg_lower_bound[neg_mask]),
            torch.full_like(r[neg_mask], 1.0 + base_eps),
        )

        policy_loss_1 = mb_advantages * r
        policy_loss_2 = mb_advantages * clipped_r
        return -torch.min(policy_loss_1, policy_loss_2).mean()

    def _opspo_fixed_loss(self, ratios, mb_advantages):
        """
        OPSPO variant with fixed threshold for tail-cap construction:
        default epsilon is induced by the fixed threshold in each group.
        """
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed_threshold_epsilon(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed_threshold_epsilon(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPSPO_FIXED: group too small for tail caps; falling back to SPO.')
            return self._spo_loss(r, mb_advantages)

        plus_sorted, plus_order, q_plus, plus_caps, plus_threshold = plus_res
        minus_sorted, minus_order, q_minus, minus_caps, minus_threshold = minus_res

        plus_default_eps = max(abs(float(plus_threshold) - 1.0), 1e-8)
        minus_threshold_safe = max(float(minus_threshold), 1e-8)
        minus_default_eps = max(abs(1.0 - 1.0 / minus_threshold_safe), 1e-8)
        eps = torch.full_like(r, plus_default_eps)
        eps[idx_minus] = minus_default_eps

        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t - 1)

        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1 - minus_floor_t)

        #eps = torch.clamp(eps, min=selpf.args.epsilon)
        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _opspo_fixed_adv_loss(self, ratios, mb_advantages):
        """
        OPSPO_FIXED variant where tail targets are assigned by advantage magnitude
        within each sign group instead of the current ratio ordering.
        """
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        plus_res = self._tail_caps_fixed_threshold_epsilon(r_safe, idx_plus, group_name='positive')
        minus_res = self._tail_caps_fixed_threshold_epsilon(1.0 / r_safe, idx_minus, group_name='negative')

        if plus_res is None or minus_res is None:
            warnings.warn('OPSPO_FIXED_ADV: group too small for tail caps; falling back to SPO.')
            return self._spo_loss(r, mb_advantages)

        _, _, q_plus, plus_caps, plus_threshold = plus_res
        _, _, q_minus, minus_caps, minus_threshold = minus_res

        plus_default_eps = max(abs(float(plus_threshold) - 1.0), 1e-8)
        minus_threshold_safe = max(float(minus_threshold), 1e-8)
        minus_default_eps = max(abs(1.0 - 1.0 / minus_threshold_safe), 1e-8)
        eps = torch.full_like(r, plus_default_eps)
        eps[idx_minus] = minus_default_eps

        plus_adv = torch.abs(mb_advantages[idx_plus])
        plus_adv_order = torch.argsort(plus_adv)
        plus_tail = plus_adv_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t - 1)

        minus_adv = torch.abs(mb_advantages[idx_minus])
        minus_adv_order = torch.argsort(minus_adv)
        minus_tail = minus_adv_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1 - minus_floor_t)

        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

    def _compute_annealed_k(self, current_update, total_updates):
        """
        Piecewise-linear schedule for k (GPD shape):
        - progress in [0, 0.5]: 0.7 -> 0.49
        - progress in (0.5, 1.0]: 0.49 -> 0.33
        """
        if current_update is None or total_updates is None or total_updates <= 1:
            return 0.7

        progress = (float(current_update) - 1.0) / (float(total_updates) - 1.0)
        progress = float(np.clip(progress, 0.0, 1.0))

        if progress <= 0.5:
            local = progress / 0.5
            return 0.7 + (0.49 - 0.7) * local

        local = (progress - 0.5) / 0.5
        return 0.49 + (0.33 - 0.49) * local

    def _opspo_fixed_anneal_loss(self, ratios, mb_advantages, global_step=None, current_update=None, total_updates=None):
        """
        Same as OPSPO_FIXED, but anneals GPD shape k from:
        0.7 -> 0.49 (first half of training), then 0.49 -> 0.33 (second half).
        """
        r = ratios
        r_safe = r.clamp_min(1e-8)
        device = r.device
        dtype = r.dtype

        idx_plus = torch.nonzero(mb_advantages >= 0, as_tuple=False).squeeze(-1)
        idx_minus = torch.nonzero(mb_advantages < 0, as_tuple=False).squeeze(-1)

        k = self._compute_annealed_k(current_update=current_update, total_updates=total_updates)
        if global_step is not None:
            self.writer.add_scalar('charts/opspo_fixed_anneal_k', k, global_step)

        plus_res = self._tail_caps_fixed_threshold_epsilon(
            r_safe, idx_plus, group_name='positive', shape_override=k
        )
        minus_res = self._tail_caps_fixed_threshold_epsilon(
            1.0 / r_safe, idx_minus, group_name='negative', shape_override=k
        )

        if plus_res is None or minus_res is None:
            warnings.warn('OPSPO_FIXED_ANNEAL: group too small for tail caps; falling back to SPO.')
            return self._spo_loss(r, mb_advantages)

        plus_sorted, plus_order, q_plus, plus_caps, plus_threshold = plus_res
        minus_sorted, minus_order, q_minus, minus_caps, minus_threshold = minus_res

        plus_default_eps = max(abs(float(plus_threshold) - 1.0), 1e-8)
        minus_threshold_safe = max(float(minus_threshold), 1e-8)
        minus_default_eps = max(abs(1.0 - 1.0 / minus_threshold_safe), 1e-8)
        eps = torch.full_like(r, plus_default_eps)
        eps[idx_minus] = minus_default_eps

        plus_tail = plus_order[q_plus:]
        plus_idx = idx_plus[plus_tail]
        plus_caps_t = torch.tensor(plus_caps, device=device, dtype=dtype)
        eps[plus_idx] = torch.abs(plus_caps_t - 1)

        minus_tail = minus_order[q_minus:]
        minus_idx = idx_minus[minus_tail]
        minus_floor_t = torch.tensor(1.0 / minus_caps, device=device, dtype=dtype)
        eps[minus_idx] = torch.abs(1 - minus_floor_t)

        pos_mask = mb_advantages >= 0
        neg_mask = ~pos_mask
        penalty = torch.zeros_like(r)
        penalty[pos_mask] = torch.pow(r[pos_mask] - 1, 2) / (2 * eps[pos_mask])
        penalty[neg_mask] = torch.pow(r[neg_mask] - 1, 2) / (2 * eps[neg_mask])

        return -(mb_advantages * r - torch.abs(mb_advantages) * penalty).mean()

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
        m = int(min(0.1 * s, 1.5 * np.sqrt(s)))#int(min(0.2 * s, 3 * np.sqrt(s)))
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
        scale = 1.0
        shape = 0.7
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
        caps = threshold + q_excess
        #caps = np.maximum(threshold + q_excess, self.args.epsilon)
        #caps = np.minimum(caps, max_tail)

        #caps = threshold+q_excess
        #caps = np.maximum(threshold + q_excess, self.args.epsilon)
        #print(caps)
        #caps = np.minimum(threshold + q_excess, max_tail-1.0)

        #print(caps)
        return (vals_sorted, order, q, caps)

    def _tail_caps_fixed_threshold_epsilon(self, values, idx, group_name, shape_override=None):
        """
        Same as _tail_caps_fixed, but threshold is fixed from default epsilon.
        Tail membership still uses order statistics via q = s - m.
        """
        if idx.numel() < 2:
            warnings.warn(f'OPSPO_FIXED: {group_name} group too small for tail caps (n={idx.numel()}).')
            return None

        vals = values[idx]
        vals_sorted, order = vals.sort()
        s = vals_sorted.numel()
        m = int(min(0.2 * s, 3 * np.sqrt(s)))
        if m < 1 or s - m < 1:
            warnings.warn(f'OPSPO_FIXED: {group_name} tail size too small (n={s}, m={m}).')
            return None

        q = s - m

        max_tail = vals_sorted[-1].item()

        shape = float(shape_override if shape_override is not None else self.args.gpd_shape)
        scale = float(self.args.gpd_scale)
        threshold = 1.0 + float(self.args.epsilon) #- float(m / s * scale / (1.0 - shape))
        
        p = (np.arange(1, m + 1) - 0.5) / m
        q_excess = genpareto.ppf(p, shape, loc=0, scale=scale)
        q_excess = np.nan_to_num(
            q_excess,
            nan=max_tail - threshold,
            posinf=max_tail - threshold,
            neginf=0.0,
        )
        caps = threshold + q_excess
        #caps = np.minimum(caps, max_tail)

        return (vals_sorted, order, q, caps, threshold)
