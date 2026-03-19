import argparse
import copy
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import Agent
from trainer import Trainer

ALGO_ALIASES = {
    'opo-fixed-loss': 'opspo_fixed',
    'opo-fixed-adv-loss': 'opspo_fixed_adv',
}
ALGO_CHOICES = [
    'ppo', 'ppo_lambda', 'tr-ppo', 'spo', 'opo', 'opo-penalty', 'opspo', 'opspo_naive', 'oppo_ranked_clip',
    'opspo_fixed', 'opspo_fixed_adv', 'opspo_fixed_anneal',
    *ALGO_ALIASES.keys(),
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=ALGO_CHOICES, default='spo')
    parser.add_argument('--envs', type=str, default=None, help='Comma-separated MuJoCo env IDs')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated random seeds')
    parser.add_argument('--policy_layers', type=int, choices=[3, 7], default=3)
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--torch_deterministic', type=bool, default=True)
    parser.add_argument('--total_time_steps', type=int, default=int(1e7))
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--learning_rate_decay', type=bool, default=True)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--mini_batches', type=int, choices=[1, 4, 32], default=4)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--advantage_normalization', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--initial_epsilon', type=float, default=None)
    parser.add_argument('--lambda_0', type=float, default=1.0)
    parser.add_argument('--kappa_0', type=float, default=0.8)
    parser.add_argument('--gpd_shape', type=float, default=0.49)
    parser.add_argument('--gpd_scale', type=float, default=0.5)
    parser.add_argument('--opspo_tail_only_penalty', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    # This is for PPO
    parser.add_argument('--adaptive_learning_rate', type=bool, default=False)
    parser.add_argument('--desired_kl', type=float, default=0.01)
    args = parser.parse_args()
    if args.initial_epsilon is None:
        args.initial_epsilon = args.epsilon
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.mini_batches)
    return args


def normalize_algo_name(algo):
    return ALGO_ALIASES.get(algo, algo)


def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        return env
    return thunk


def main(env_id, seed, algo, base_args=None):
    args = copy.deepcopy(base_args) if base_args is not None else get_args()
    args.env_id = env_id
    args.seed = seed
    args.algo = normalize_algo_name(algo or args.algo)
    if args.algo == 'opspo_fixed_adv':
        # Force full-batch updates for this variant: one minibatch per epoch.
        args.minibatch_size = args.batch_size
        args.mini_batches = 1

    # Adaptive learning rate
    if args.adaptive_learning_rate and args.algo == 'ppo':
        args.learning_rate_decay = False

    # Different algorithms
    run_name = (
            args.algo + '_' + str(args.epsilon) +
            '_layers_' + str(args.policy_layers) +
            '_mini_bs_' + str(args.minibatch_size) +
            '_seed_' + str(args.seed)
    )
    if args.adaptive_learning_rate and args.algo == 'ppo':
        run_name += '_adaptive_lr'
    assert args.algo in ALGO_CHOICES, 'wrong algorithm name'
    print('[algorithm:', args.algo + ']', '[env:', args.env_id + ']', '[seed:', str(args.seed) + ']')

    # Save training logs
    path_string = str(args.env_id) + '/' + run_name
    writer = SummaryWriter(path_string)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # Initialize environments
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.gamma) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), 'only continuous action space is supported'

    # Initialize agent
    agent = Agent(envs, args.policy_layers).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)
    trainer = Trainer(args, agent, optimizer, writer)

    # Initialize buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    mean = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    std = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # Data collection
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_time_steps // args.batch_size

    # This is for plotting
    update_index = 1
    episodic_returns = []

    for update in tqdm(range(1, num_updates + 1)):

        # Linear decay of learning rate
        if args.learning_rate_decay:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value, mean_std = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # Mean and standard deviation
            mean[step] = mean_std.loc
            std[step] = mean_std.scale

            # Update the environments
            next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if 'final_info' not in info:
                continue

            for item in info['final_info']:
                if item is None:
                    continue
                writer.add_scalar('charts/episodic_return', item['episode']['r'][0], global_step)

                # This is for plotting
                if update == update_index:
                    episodic_returns.append(item['episode']['r'][0])
                else:
                    writer.add_scalar(
                        'This is for plotting/average_return', np.mean(episodic_returns), update_index
                    )
                    episodic_returns.clear()
                    episodic_returns.append(item['episode']['r'][0])
                    update_index += 1

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)):
                next_non_terminal = 1.0 - next_done if t == args.num_steps - 1 else 1.0 - dones[t + 1]
                next_values = next_value if t == args.num_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            returns = advantages + values

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mean = mean.reshape(args.batch_size, -1)
        b_std = std.reshape(args.batch_size, -1)

        # Train
        trainer.train(
            global_step,
            b_obs,
            b_actions,
            b_log_probs,
            b_advantages,
            b_returns,
            b_values,
            b_mean,
            b_std,
            current_update=update,
            total_updates=num_updates,
        )

        # Save the data during the training process
        y_pre, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pre) / var_y
        writer.add_scalar('losses/explained_variance', explained_var, global_step)
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


def _parse_csv_list(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def run(algo, env_ids=None, seeds=None, base_args=None):
    """
    Choose the environments and random seeds
    """
    if env_ids is None:
        env_ids = [
            # 'Ant-v4',
            # 'HalfCheetah-v4',
            # 'Hopper-v4',
            'Humanoid-v4',
            # 'HumanoidStandup-v4',
            # 'Walker2d-v4'
        ]
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    for env_id in env_ids:
        for seed in seeds:
            main(env_id, seed, algo, base_args=base_args)


if __name__ == '__main__':
    # ppo, spo, opo, opspo, or opspo_fixed
    args = get_args()
    envs = _parse_csv_list(args.envs) if args.envs else None
    seeds = [int(s) for s in _parse_csv_list(args.seeds)] if args.seeds else None
    run(args.algo, env_ids=envs, seeds=seeds, base_args=args)
