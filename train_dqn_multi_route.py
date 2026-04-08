import os
import random
import glob
import argparse
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci

from rl_environment import TrafficEnvironment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save replay buffer to file"""
        save_dir = os.path.dirname(filepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Saved replay buffer ({len(self.buffer)} samples) to {filepath}")
    
    def load(self, filepath: str, max_samples: int = None):
        """Load replay buffer from file"""
        if not os.path.exists(filepath):
            print(f"Warning: Replay buffer file '{filepath}' not found.")
            return

        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)

        # Cap the number of samples if requested
        if max_samples is not None and len(loaded_buffer) > max_samples:
            loaded_buffer = random.sample(loaded_buffer, max_samples)

        # Append to current buffer (deque respects maxlen capacity)
        for sample in loaded_buffer:
            self.buffer.append(sample)
        
        print(f"Loaded {len(loaded_buffer)} samples from {filepath} into replay buffer")
        print(f"Total buffer size: {len(self.buffer)}")
    
    def merge(self, other_buffer):
        """Merge another replay buffer into this one"""
        for sample in other_buffer.buffer:
            self.buffer.append(sample)
        print(f"Merged buffer. Total size: {len(self.buffer)}")


def flatten_obs(obs):
    """
    env.reset() in rl_environment.py returns a tuple of arrays,
    while env.step() returns a pre-concatenated 1-D array.
    This helper normalises both cases to a flat 1-D numpy array.
    """
    if isinstance(obs, (list, tuple)):
        return np.concatenate(obs, axis=0).astype(np.float32)
    return np.array(obs, dtype=np.float32)


def find_route_files(route_dir: str):
    """
    Finds all route files (.rou.xml) inside a directory.
    Returns a sorted list of full file paths.
    """
    if not os.path.exists(route_dir):
        raise ValueError(f"Route directory '{route_dir}' does not exist")

    pattern = os.path.join(route_dir, "*.rou.xml")
    route_files = glob.glob(pattern)

    if not route_files:
        raise ValueError(f"No .rou.xml files found in directory '{route_dir}'")

    route_files.sort()  # Sort for deterministic ordering
    return route_files


def train_dqn_multi_route(
    num_episodes: int = 500,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 300,
    target_update_interval: int = 10,
    model_save_path: str = "Pretrain/dqn_traffic.pt",
    pretrain_path: str = None,
    start_episode: int = 0,
    route_dir: str = None,
    route_file: str = None,
    config_file: str = 'SumoCfg/cross.sumocfg',
    gui: bool = False,
    step_length: float = 0.1,
    delay: int = 0,
    previous_buffer_path: str = None,
    save_buffer_path: str = None,
    buffer_retention_ratio: float = 0.3,
    route_selection_mode: str = 'random',  # 'random', 'cycle', or 'block'
    episodes_per_route: int = 3,  # episodes per route when using 'block' mode
):
    """
    Train a DQN agent across multiple route files.

    Args:
        route_dir: Directory containing route files (.rou.xml). When provided,
                   training cycles over all files in the directory.
        route_file: Single route file path (overrides route_dir when set).
        route_selection_mode: 'random' — pick a random file each episode;
                              'cycle'  — iterate through files in order;
                              'block'  — train N episodes per route then advance.
        episodes_per_route: Number of episodes per route file in 'block' mode.
    """
    # Resolve route file list
    if route_file:
        route_files = [route_file]
        print(f"Using single route file: {route_file}")
    elif route_dir:
        route_files = find_route_files(route_dir)
        print(f"Found {len(route_files)} route files in '{route_dir}':")
        for i, rf in enumerate(route_files, 1):
            print(f"  {i}. {os.path.basename(rf)}")
    else:
        # No route specified — fall back to the route defined in the SUMO config
        route_files = [None]
        print("Using default route file from config")

    print(f"Using device: {DEVICE}")
    print("Training hyperparameters:")
    print(f"  num_episodes={num_episodes}")
    print(f"  buffer_capacity={buffer_capacity}")
    print(f"  batch_size={batch_size}")
    print(f"  gamma={gamma}")
    print(f"  lr={lr}")
    print(f"  eps_start={eps_start}")
    print(f"  eps_end={eps_end}")
    print(f"  eps_decay_episodes={eps_decay_episodes}")
    print(f"  target_update_interval={target_update_interval}")
    print(f"  model_save_path='{model_save_path}'")
    print(f"  env: gui={gui}, step_length={step_length}, delay={delay}")
    print(f"  config_file='{config_file}'")
    print(f"  route_selection_mode='{route_selection_mode}'")
    if route_selection_mode == 'block':
        print(f"  episodes_per_route={episodes_per_route}")

    # Ensure the model save directory exists
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Build a temporary environment to infer observation/action dimensions
    initial_route = route_files[0] if route_files[0] else None
    env = TrafficEnvironment(
        config_file=config_file,
        step_length=step_length,
        gui=gui,
        delay=delay,
        route_file=initial_route
    )

    obs, _ = env.reset()
    obs = flatten_obs(obs)
    env._close_sumo()

    input_dim = obs.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(input_dim, n_actions).to(DEVICE)
    target_net = DQN(input_dim, n_actions).to(DEVICE)
    
    # Load pretrained weights if provided
    if pretrain_path is not None and os.path.exists(pretrain_path):
        print(f"Loading pretrained model from {pretrain_path}")
        # weights_only=False for compatibility with PyTorch 2.6+ (checkpoints may contain numpy objects)
        policy_net.load_state_dict(torch.load(pretrain_path, map_location=DEVICE, weights_only=False))
        target_net.load_state_dict(policy_net.state_dict())
        print("Pretrained model loaded successfully!")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        if pretrain_path is not None:
            print(f"Warning: Pretrain path '{pretrain_path}' not found. Starting from scratch.")
    
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Pre-load old replay buffer to mitigate catastrophic forgetting
    if previous_buffer_path is not None:
        retention_samples = int(buffer_capacity * buffer_retention_ratio)
        replay_buffer.load(previous_buffer_path, max_samples=retention_samples)
        print(f"Loaded previous replay buffer to prevent catastrophic forgetting")
        print(f"  Retention ratio: {buffer_retention_ratio} ({retention_samples} samples)")

    # Track best reward for best-model checkpointing
    best_reward = float('-inf')
    best_episode = 0
    base, ext = os.path.splitext(model_save_path)
    best_model_path = f"{base}_best{ext}"

    # Per-route usage statistics
    route_stats = {rf: {'count': 0, 'total_reward': 0.0} for rf in route_files}

    def epsilon_by_episode(episode_idx: int) -> float:
        # Linear decay from eps_start to eps_end over eps_decay_episodes
        # Computed against the global episode index (start_episode + episode_idx)
        total_episode = start_episode + episode_idx
        frac = min(1.0, total_episode / max(1, eps_decay_episodes))
        return eps_start + frac * (eps_end - eps_start)
    
    def select_route(episode_idx: int):
        """Select the route file for the current episode."""
        if len(route_files) == 1:
            return route_files[0]

        if route_selection_mode == 'random':
            return random.choice(route_files)
        elif route_selection_mode == 'cycle':
            return route_files[episode_idx % len(route_files)]
        elif route_selection_mode == 'block':
            # Train N episodes per route, then advance to the next one.
            # Example: episodes_per_route=3, 5 routes →
            #   episodes 0-2: route[0], 3-5: route[1], 6-8: route[2], …
            block_index = episode_idx // episodes_per_route
            route_index = block_index % len(route_files)
            return route_files[route_index]
        else:
            raise ValueError(f"Unknown route_selection_mode: {route_selection_mode}")

    # A fresh environment is created every episode to allow route switching
    env = None
    previous_route = None

    for episode in range(num_episodes):
        current_episode = start_episode + episode

        selected_route = select_route(episode)
        route_stats[selected_route]['count'] += 1

        # Announce block transitions in 'block' mode
        if route_selection_mode == 'block' and selected_route != previous_route:
            route_name = os.path.basename(selected_route) if selected_route else "default"
            block_num = (episode // episodes_per_route) + 1
            print(f"\n>>> Starting block {block_num}: Training on route '{route_name}' (episodes {episode+1}-{min(episode+episodes_per_route, num_episodes)}) <<<")
            previous_route = selected_route

        # Close the previous environment
        if env is not None:
            try:
                env._close_sumo()
            except:
                pass

        # Create a new environment with the selected route file
        env = TrafficEnvironment(
            config_file=config_file,
            step_length=step_length,
            gui=gui,
            delay=delay,
            route_file=selected_route
        )
        
        obs, _ = env.reset()
        state = flatten_obs(obs)

        episode_reward = 0.0
        done = False

        while not done:
            eps = epsilon_by_episode(episode)

            # --- Action masking: only allow phases with vehicles present ---
            valid_actions = env._get_valid_actions()

            if random.random() < eps:
                # Exploration: sample uniformly from valid actions only
                action = random.choice(valid_actions)
            else:
                # Exploitation: mask Q-values of invalid actions with -inf
                with torch.no_grad():
                    state_t = torch.tensor(
                        state, dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                    q_values = policy_net(state_t).squeeze(0)

                    mask = torch.full(
                        (env.action_space.n,), float('-inf'), device=DEVICE
                    )
                    for a in valid_actions:
                        mask[a] = q_values[a]

                    action = int(torch.argmax(mask).item())

            next_obs, reward, done, _ = env.step(action)
            next_state = flatten_obs(next_obs)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Train step
            if len(replay_buffer) >= batch_size:
                (
                    states_b,
                    actions_b,
                    rewards_b,
                    next_states_b,
                    dones_b,
                ) = replay_buffer.sample(batch_size)

                # Q(s,a)
                q_values = policy_net(states_b).gather(
                    1, actions_b.unsqueeze(1)
                ).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(1)[0]
                    target_q = rewards_b + gamma * next_q_values * (1 - dones_b)

                loss = nn.functional.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        route_stats[selected_route]['total_reward'] += episode_reward

        # Sync target network at the configured interval
        if (current_episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Fetch SUMO simulation time at episode end
        try:
            sim_time = traci.simulation.getTime()
        except traci.TraCIException:
            sim_time = None

        # Save best model whenever a new highest reward is achieved
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = current_episode + 1
            best_dir = os.path.dirname(best_model_path)
            if best_dir and not os.path.exists(best_dir):
                os.makedirs(best_dir, exist_ok=True)
            torch.save(policy_net.state_dict(), best_model_path)
            print(f"*** New best model! Reward: {best_reward:.2f} (Episode {best_episode}) ***")

        route_name = os.path.basename(selected_route) if selected_route else "default"
        if sim_time is not None:
            print(
                f"Episode {current_episode + 1} (local: {episode + 1}/{num_episodes}) | "
                f"route={route_name} | "
                f"epsilon={eps:.3f} | reward={episode_reward:.2f} | "
                f"best_reward={best_reward:.2f} (ep {best_episode}) | "
                f"buffer={len(replay_buffer)} | "
                f"sim_time_end={sim_time:.1f}s"
            )
        else:
            print(
                f"Episode {current_episode + 1} (local: {episode + 1}/{num_episodes}) | "
                f"route={route_name} | "
                f"epsilon={eps:.3f} | reward={episode_reward:.2f} | "
                f"best_reward={best_reward:.2f} (ep {best_episode}) | "
                f"buffer={len(replay_buffer)}"
            )

        # Save a periodic checkpoint every 50 episodes (never overwrites previous)
        if (current_episode + 1) % 50 == 0:
            base, ext = os.path.splitext(model_save_path)
            ckpt_path = f"{base}__ep{current_episode + 1}{ext}"
            ckpt_dir = os.path.dirname(ckpt_path)
            if ckpt_dir and not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(policy_net.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Close the last environment
    if env is not None:
        try:
            env._close_sumo()
        except:
            pass

    # Save the final model
    final_dir = os.path.dirname(model_save_path)
    if final_dir and not os.path.exists(final_dir):
        os.makedirs(final_dir, exist_ok=True)
    torch.save(policy_net.state_dict(), model_save_path)
    print(f"Training finished. Final model saved to {model_save_path}")
    print(f"Best model (reward={best_reward:.2f}, episode={best_episode}) saved to {best_model_path}")
    
    print("\n" + "="*80)
    print("ROUTE FILES USAGE STATISTICS")
    print("="*80)
    for route_file, stats in route_stats.items():
        route_name = os.path.basename(route_file) if route_file else "default"
        avg_reward = stats['total_reward'] / stats['count'] if stats['count'] > 0 else 0.0
        print(f"{route_name}:")
        print(f"  Episodes: {stats['count']} ({100.0 * stats['count'] / num_episodes:.1f}%)")
        print(f"  Total reward: {stats['total_reward']:.2f}")
        print(f"  Average reward: {avg_reward:.2f}")
    print("="*80)
    
    # Optionally persist the replay buffer for the next training session
    if save_buffer_path is not None:
        replay_buffer.save(save_buffer_path)


def find_latest_checkpoint(base_name="dqn_traffic", folder: str = "Pretrain"):
    """Find the most recent checkpoint in a folder based on episode number."""
    pattern = os.path.join(folder, f"{base_name}__ep*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    # Extract episode numbers and find max
    def extract_episode(path):
        try:
            # Extract number from "dqn_traffic__epXXX.pt"
            name = os.path.basename(path)
            ep_str = name.split("__ep")[1].split(".")[0]
            return int(ep_str)
        except:
            return -1
    
    latest = max(checkpoints, key=extract_episode)
    episode_num = extract_episode(latest)
    return latest, episode_num


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DQN for traffic signal control using multiple route files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--num-episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--buffer-capacity', type=int, default=500000,
                        help='Replay buffer capacity')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='Final epsilon for epsilon-greedy')
    parser.add_argument('--eps-decay-episodes', type=int, default=300,
                        help='Number of episodes for epsilon decay')
    parser.add_argument('--target-update-interval', type=int, default=10,
                        help='Interval to update target network')
    
    # Model paths
    parser.add_argument('--save-folder', type=str, default='Pretrain',
                        help='Folder to save models')
    parser.add_argument('--model-name', type=str, default='dqn_traffic.pt',
                        help='Name of the model file to save')
    
    # Pretrain options
    parser.add_argument('--load-pretrain', action='store_true',
                        help='Whether to load pretrained model')
    parser.add_argument('--pretrain-path', type=str, default=None,
                        help='Path to pretrained model file (required if --load-pretrain is set)')
    parser.add_argument('--start-episode', type=int, default=0,
                        help='Starting episode number (for epsilon decay calculation)')
    
    # Environment options
    parser.add_argument('--route-dir', type=str, default=None,
                        help='Directory containing route files (.rou.xml). All files in this directory will be used for training.')
    parser.add_argument('--route-file', type=str, default=None,
                        help='Single route file path (e.g., SumoCfg/train/balance.rou.xml). If specified, --route-dir will be ignored.')
    parser.add_argument('--route-selection-mode', type=str, default='random',
                        choices=['random', 'cycle', 'block'],
                        help='How to select route files: "random" for random selection each episode, "cycle" for sequential cycling, "block" for training N episodes on each route before moving to next')
    parser.add_argument('--episodes-per-route', type=int, default=3,
                        help='Number of episodes to train on each route file when using "block" mode (default: 3)')
    parser.add_argument('--config-file', type=str, default='SumoCfg/cross.sumocfg',
                        help='SUMO config file path')
    parser.add_argument('--gui', action='store_true',
                        help='Show SUMO GUI during training')
    parser.add_argument('--step-length', type=float, default=0.1,
                        help='SUMO step length in seconds')
    parser.add_argument('--delay', type=int, default=0,
                        help='SUMO GUI delay in milliseconds')
    
    # Catastrophic Forgetting prevention
    parser.add_argument('--previous-buffer', type=str, default=None,
                        help='Path to a replay buffer from a previous training run (prevents catastrophic forgetting)')
    parser.add_argument('--save-buffer', type=str, default=None,
                        help='Path to save the replay buffer after training (for use in a future session)')
    parser.add_argument('--buffer-retention-ratio', type=float, default=0.3,
                        help='Fraction of old buffer samples to retain (0.0–1.0, default: 0.3)')
    
    return parser.parse_args()


def resolve_pretrain_path(args):
    """
    Resolve pretrain path based on arguments.
    Returns (pretrain_path, start_episode)
    """
    if not args.load_pretrain:
        return None, 0
    
    # pretrain path is required when --load-pretrain is set
    if not args.pretrain_path:
        raise ValueError("--pretrain-path is required when --load-pretrain is set")
    
    if not os.path.exists(args.pretrain_path):
        raise FileNotFoundError(f"Pretrain path '{args.pretrain_path}' not found.")
    
    print(f"Loading pretrained model from: {args.pretrain_path}")
    return args.pretrain_path, args.start_episode


if __name__ == "__main__":
    args = parse_args()
    
    # Resolve pretrain path
    pretrain_path, start_episode = resolve_pretrain_path(args)
    
    # Build model save path
    model_save_path = os.path.join(args.save_folder, args.model_name)
    
    # Print configuration
    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Save folder: {args.save_folder}")
    print(f"Model name: {args.model_name}")
    print(f"Load pretrain: {args.load_pretrain}")
    if pretrain_path:
        print(f"Pretrain path: {pretrain_path}")
        print(f"Start episode: {start_episode}")
    if args.route_file:
        print(f"Route file: {args.route_file}")
    elif args.route_dir:
        print(f"Route directory: {args.route_dir}")
    else:
        print(f"Route: None (using default from config)")
    print(f"Route selection mode: {args.route_selection_mode}")
    if args.route_selection_mode == 'block':
        print(f"Episodes per route: {args.episodes_per_route}")
    print(f"Config file: {args.config_file}")
    print(f"Step length: {args.step_length}")
    print(f"Delay: {args.delay}")
    print("="*80)
    
    train_dqn_multi_route(
        num_episodes=args.num_episodes,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        target_update_interval=args.target_update_interval,
        model_save_path=model_save_path,
        pretrain_path=pretrain_path,
        start_episode=start_episode,
        route_dir=args.route_dir,
        route_file=args.route_file,
        config_file=args.config_file,
        gui=args.gui,
        step_length=args.step_length,
        delay=args.delay,
        previous_buffer_path=args.previous_buffer,
        save_buffer_path=args.save_buffer,
        buffer_retention_ratio=args.buffer_retention_ratio,
        route_selection_mode=args.route_selection_mode,
        episodes_per_route=args.episodes_per_route,
    )
