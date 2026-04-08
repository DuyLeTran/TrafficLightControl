import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import multiprocessing as mp
import queue
import numpy as np
import torch
import traci
import socket
import random
import subprocess
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Tkinter compatibility
import matplotlib.pyplot as plt
from sumolib import checkBinary

from rl_environment import TrafficEnvironment
from train_dqn_multi_route import DQN, flatten_obs, DEVICE


def find_free_port(start_port=8013, max_attempts=500):
    """Find a free port for SUMO to use"""
    for _ in range(max_attempts):
        port = random.randint(8013, 9000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find free port")


def run_baseline_worker(config_file, route_file, metrics_queue, stop_event, start_event):
    """Worker function for baseline - runs in separate process"""
    try:
        # Find free port for baseline
        port = find_free_port()
        print(f"Baseline using port: {port}")
        metrics_queue.put(("baseline_port", port))

        env = TrafficEnvironment(
            config_file=config_file,
            gui=True,
            route_file=route_file,
            step_length=1
        )

        # Override reset to use unique port and NOT auto-start
        original_init_sumo = env._init_sumo

        def reset_with_port(seed=None, options=None):
            """Custom reset function: ensure SUMO uses unique port and does NOT auto-start"""
            # Close old SUMO instance
            env._close_sumo()

            # Get args from original _init_sumo
            args = original_init_sumo()
            sumo_binary = args[0]
            sumo_args = args[1:]

            # IMPORTANT: Add --remote-port to sumo_args
            if "--remote-port" not in sumo_args:
                sumo_args = sumo_args + ["--remote-port", str(port)]

            # Start SUMO process directly using subprocess
            process = subprocess.Popen(
                [sumo_binary] + sumo_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for SUMO to start
            time.sleep(0.5)

            # Connect to SUMO using traci.init() with specified port
            traci.init(port)

            # Reset environment state variables
            env.current_step = 0
            env.current_phase = traci.trafficlight.getPhase(env.tls_id)
            env.rt = 0.0

            # Initialize previous values
            prev_queue = env._get_queue_length()
            prev_vehicle = env._get_vehicle_count()
            env.prev_queue_lengths = np.array(prev_queue, dtype=np.float32)
            env.prev_vehicle_counts = np.array(prev_vehicle, dtype=np.float32)

            # Get initial observation
            observation = env._get_observation()
            return observation, {}

        env.reset = reset_with_port

        # Reset environment with port (retry multiple times if fails)
        max_retries = 3
        for retry in range(max_retries):
            try:
                obs, _ = env.reset()
                break
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"Baseline: Failed to initialize after {max_retries} retries: {e}")
                time.sleep(0.5 * (retry + 1))

        # Wait for signal to start simulation (when "Run" button is pressed)
        print("Baseline: Waiting for signal to start...")
        start_event.wait()
        print("Baseline: Received signal, starting simulation!")

        # Call simulationStep() first time to start simulation
        try:
            traci.simulationStep()
            print("Baseline: Called simulationStep() first time")
        except Exception as e:
            print(f"Baseline: Error calling simulationStep() first time: {e}")

        current_phase = traci.trafficlight.getPhase(env.tls_id)
        prev_phase = current_phase

        prev_queue = np.array(env._get_queue_length(), dtype=np.float32)
        prev_vehicle = np.array(env._get_vehicle_count(), dtype=np.float32)
        rt = 0.0
        phase_start_time = 0.0  # Track when the current phase started

        done = False
        duration_green = 5

        while not done and not stop_event.is_set():
            try:
                if traci.simulation.getMinExpectedNumber() == 0:
                    done = True
                    break
            except Exception:
                pass

            try:
                env._run_step(duration_green)
            except Exception as e:
                print(f"Baseline step error: {e}")
                break

            try:
                current_phase = traci.trafficlight.getPhase(env.tls_id)
            except Exception:
                break

            changed_phase = (current_phase != prev_phase)

            # Track phase duration: when phase changes, record duration of the old phase
            if changed_phase and prev_phase in env.green_phases:
                # Old phase duration = rt accumulated before this step + the 5 s just executed
                phase_duration = rt + duration_green
                if phase_duration > 0:
                    metrics_queue.put(("baseline_phase_duration", {
                        "phase": prev_phase,
                        "duration": phase_duration
                    }))
                rt = 0.0
                phase_start_time = traci.simulation.getTime()
            elif changed_phase and current_phase in env.green_phases:
                # Switched to a new green phase — reset timer
                rt = 0.0
                phase_start_time = traci.simulation.getTime()
            elif current_phase in env.green_phases:
                # Still in the same green phase — accumulate rt
                rt += duration_green

            try:
                observation = env._get_observation()
                current_queue = np.array(env._get_queue_length(), dtype=np.float32)
                current_vehicle = np.array(env._get_vehicle_count(), dtype=np.float32)
            except Exception:
                break

            phase_unchanged = env._check_phase_unchanged(
                prev_queue, prev_vehicle,
                current_queue, current_vehicle,
                current_phase
            )

            reward = env._calculate_reward(observation, changed_phase, rt > 120, phase_unchanged)

            try:
                queue_total = sum(env._get_queue_length())
                waiting_total = sum(env._get_max_accumulated_waiting_time_per_lane())
                sim_time = traci.simulation.getTime()
            except Exception:
                break

            metrics_queue.put(("baseline", {
                "queue": queue_total,
                "waiting": waiting_total,
                "reward": reward,
                "sim_time": sim_time,
                "phase_unchanged": phase_unchanged
            }))

            prev_queue = current_queue
            prev_vehicle = current_vehicle
            prev_phase = current_phase
            
            # Record the final phase duration when the simulation ends
            if done and current_phase in env.green_phases:
                phase_duration = rt
                metrics_queue.put(("baseline_phase_duration", {
                    "phase": current_phase,
                    "duration": phase_duration
                }))

            try:
                done = traci.simulation.getMinExpectedNumber() == 0
            except Exception:
                done = True

            if done:
                break

        metrics_queue.put(("baseline", "done"))
        env._close_sumo()

    except Exception as e:
        print(f"Baseline error: {e}")
        metrics_queue.put(("baseline_error", str(e)))


def run_agent_worker(config_file, route_file, model_path, metrics_queue, stop_event, start_event):
    """Worker function for agent - runs in separate process"""
    try:
        # Find free port for agent
        port = find_free_port()
        print(f"Agent using port: {port}")
        metrics_queue.put(("agent_port", port))

        env = TrafficEnvironment(
            config_file=config_file,
            gui=True,
            route_file=route_file,
            step_length=1
        )

        # Override reset to use unique port and NOT auto-start
        original_init_sumo = env._init_sumo

        def reset_with_port(seed=None, options=None):
            """Custom reset function: ensure SUMO uses unique port and does NOT auto-start"""
            env._close_sumo()

            args = original_init_sumo()
            sumo_binary = args[0]
            sumo_args = args[1:]

            if "--remote-port" not in sumo_args:
                sumo_args = sumo_args + ["--remote-port", str(port)]

            process = subprocess.Popen(
                [sumo_binary] + sumo_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(0.5)

            traci.init(port)

            env.current_step = 0
            env.current_phase = traci.trafficlight.getPhase(env.tls_id)
            env.rt = 0.0

            prev_queue = env._get_queue_length()
            prev_vehicle = env._get_vehicle_count()
            env.prev_queue_lengths = np.array(prev_queue, dtype=np.float32)
            env.prev_vehicle_counts = np.array(prev_vehicle, dtype=np.float32)

            observation = env._get_observation()
            return observation, {}

        env.reset = reset_with_port

        # Reset environment with port (retry multiple times if fails)
        max_retries = 3
        for retry in range(max_retries):
            try:
                obs, _ = env.reset()
                obs = flatten_obs(obs)
                input_dim = obs.shape[0]
                n_actions = env.action_space.n
                break
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"Agent: Failed to initialize after {max_retries} retries: {e}")
                time.sleep(0.5 * (retry + 1))

        # Load model (DQN v2)
        policy_net = DQN(input_dim, n_actions).to(DEVICE)
        policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        policy_net.eval()

        # Reset again to start simulation
        obs, _ = env.reset()
        state = flatten_obs(obs)
        done = False
        prev_phase = env.current_phase
        prev_rt = env.rt

        # Wait for signal to start simulation
        print("Agent: Waiting for signal to start...")
        start_event.wait()
        print("Agent: Received signal, starting simulation!")

        try:
            traci.simulationStep()
            print("Agent: Called simulationStep() first time")
        except Exception as e:
            print(f"Agent: Error calling simulationStep() first time: {e}")

        while not done and not stop_event.is_set():
            with torch.no_grad():
                state_t = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)
                q_values = policy_net(state_t).squeeze(0)

                # --- Action masking: only allow phases with vehicles present (same as training) ---
                valid_actions = env._get_valid_actions()
                
                mask = torch.full(
                    (env.action_space.n,), float('-inf'), device=DEVICE
                )
                for a in valid_actions:
                    mask[a] = q_values[a]

                action = int(torch.argmax(mask).item())

            # Snapshot phase and rt before the step (needed to compute duration on phase change)
            old_phase = env.current_phase
            old_rt = env.rt
            action_phase = action * 2  # actions 0,1,2,3 map to phases 0,2,4,6

            try:
                next_obs, reward, done, info = env.step(action)
            except Exception as e:
                print(f"Agent step error: {e}")
                break
            next_state = flatten_obs(next_obs)

            # Track phase duration: on phase change, record duration of the old phase
            current_phase = env.current_phase
            if old_phase != action_phase and old_phase in env.green_phases:
                # old_rt is the time the old phase had been active before the decision.
                # env.step() does not add 5 s to the old phase when switching,
                # but the agent observed state after 5 s, so we add 5 s here.
                phase_duration = old_rt + 5.0
                if phase_duration > 0:
                    metrics_queue.put(("agent_phase_duration", {
                        "phase": old_phase,
                        "duration": phase_duration
                    }))

            # Extract phase_unchanged from info (info is a tuple: (observation, phase_unchanged))
            phase_unchanged = info[1] if isinstance(info, tuple) and len(info) > 1 else False

            try:
                queue_total = sum(env._get_queue_length())
                waiting_total = sum(env._get_max_accumulated_waiting_time_per_lane())
                sim_time = traci.simulation.getTime()
            except Exception:
                break

            metrics_queue.put(("agent", {
                "queue": queue_total,
                "waiting": waiting_total,
                "reward": reward,
                "sim_time": sim_time,
                "phase_unchanged": phase_unchanged
            }))

            prev_phase = current_phase
            prev_rt = env.rt
            state = next_state

            # Record the final phase duration when the simulation ends
            if done and current_phase in env.green_phases:
                phase_duration = env.rt
                if phase_duration > 0:
                    metrics_queue.put(("agent_phase_duration", {
                        "phase": current_phase,
                        "duration": phase_duration
                    }))

            if done:
                break

        metrics_queue.put(("agent", "done"))
        env._close_sumo()

    except Exception as e:
        print(f"Agent error: {e}")
        metrics_queue.put(("agent_error", str(e)))


class RealTimeMetrics:
    """Store real-time metrics for each simulation"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.queue_lengths = []
        self.waiting_times = []
        self.rewards = []
        self.sim_times = []
        self.current_queue = 0.0
        self.current_waiting = 0.0
        self.current_reward = 0.0
        self.current_sim_time = 0.0
        self.current_phase_unchanged = False
        self.total_reward = 0.0
        self.step_count = 0
        self.episode = 0
        self.is_running = False
        self.is_done = False
        # Add episode_data for plot_comparison compatibility
        self.episode_data = []
        # Track green phase durations: {phase_id: [duration1, duration2, ...]}
        self.phase_durations = {0: [], 2: [], 4: [], 6: []}
        self.current_phase = None
        self.current_phase_start_time = 0.0

    def get_summary(self):
        """Convert metrics to summary format similar to MetricsCollector.get_summary()"""
        summary = {}

        if self.queue_lengths:
            summary['queue_length'] = {
                'mean': np.mean(self.queue_lengths),
                'max': np.max(self.queue_lengths),
                'total_mean': np.sum(self.queue_lengths) / len(self.queue_lengths) if len(self.queue_lengths) > 0 else 0,
                'final_mean': self.queue_lengths[-1] if len(self.queue_lengths) > 0 else 0
            }

        if self.waiting_times:
            summary['waiting_time'] = {
                'mean': np.mean(self.waiting_times),
                'max': np.max(self.waiting_times),
                'total_mean': np.sum(self.waiting_times) / len(self.waiting_times) if len(self.waiting_times) > 0 else 0,
                'final_mean': self.waiting_times[-1] if len(self.waiting_times) > 0 else 0
            }

        if self.rewards:
            summary['reward'] = {
                'total_mean': np.sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else 0,
                'mean_mean': np.mean(self.rewards),
                'min_mean': np.min(self.rewards),
                'max_mean': np.max(self.rewards)
            }

        if self.sim_times:
            summary['sim_time'] = {
                'mean': np.mean(self.sim_times),
                'max': np.max(self.sim_times),
                'min': np.min(self.sim_times),
                'all': self.sim_times.copy()
            }

        # Add episode_data for reward plotting
        if self.episode_data:
            summary['episode_data'] = self.episode_data
        else:
            # If no episode_data yet, create from current data (1 episode)
            summary['episode_data'] = [{
                'queue': self.queue_lengths.copy() if self.queue_lengths else [],
                'waiting': self.waiting_times.copy() if self.waiting_times else [],
                'reward': self.rewards.copy() if self.rewards else []
            }]

        # Calculate average green phase durations
        summary['phase_durations'] = {}
        for phase_id in [0, 2, 4, 6]:
            durations = self.phase_durations.get(phase_id, [])
            if durations:
                summary['phase_durations'][phase_id] = {
                    'mean': np.mean(durations),
                    'count': len(durations),
                    'all': durations.copy()
                }
            else:
                summary['phase_durations'][phase_id] = {
                    'mean': 0.0,
                    'count': 0,
                    'all': []
                }

        return summary


class ComparisonDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Baseline vs Agent Comparison - Real-time")
        self.root.geometry("1200x800")

        # Metrics storage
        self.baseline_metrics = RealTimeMetrics()
        self.agent_metrics = RealTimeMetrics()

        # Multiprocessing
        self.baseline_process = None
        self.agent_process = None
        self.metrics_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.start_event = mp.Event()  # Event to start both simultaneously

        self.model_path = "DQN/dqn_traffic_best.pt"
        self.baseline_port = None
        self.agent_port = None
        self.baseline_frame = None
        self.agent_frame = None

        # Initialize labels dictionary BEFORE setup_ui
        self.metrics_labels = {}
        self.summary_labels = {}

        # Flag to avoid plotting multiple times
        self.plot_triggered = False

        self.setup_ui()
        self.start_update_loop()

    def setup_ui(self):
        # Top frame - Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Model selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.model_entry = ttk.Entry(control_frame, width=50)
        self.model_entry.insert(0, self.model_path)
        self.model_entry.grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Select File", command=self.select_model).grid(row=0, column=2, padx=5)

        # Config selection
        ttk.Label(control_frame, text="Config:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.config_entry = ttk.Entry(control_frame, width=50)
        self.config_entry.insert(0, "SumoCfg/cross.sumocfg")
        self.config_entry.grid(row=1, column=1, padx=5)
        ttk.Button(control_frame, text="Select File", command=self.select_config).grid(row=1, column=2, padx=5)

        ttk.Label(control_frame, text="Route:").grid(row=2, column=0, padx=5, sticky=tk.W)
        self.route_entry = ttk.Entry(control_frame, width=50)
        self.route_entry.insert(0, "SumoCfg/test/balance.rou.xml")
        self.route_entry.grid(row=2, column=1, padx=5)
        ttk.Button(control_frame, text="Select File", command=self.select_route).grid(row=2, column=2, padx=5)

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        self.start_btn = ttk.Button(button_frame, text="Load Simulation", command=self.start_comparison)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.run_btn = tk.Button(button_frame, text="Run", command=self.run_simulations,
                                  font=("Arial", 10, "bold"), bg="#2196F3", fg="white",
                                  relief=tk.RAISED, padx=10, pady=5, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_comparison, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset_metrics)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        # Main comparison frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Baseline frame
        self.baseline_frame = ttk.LabelFrame(main_frame, text="BASELINE (Port: --)", padding="10")
        self.baseline_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.setup_metrics_display(self.baseline_frame, "baseline")

        # Agent frame
        self.agent_frame = ttk.LabelFrame(main_frame, text="AGENT (Port: --)", padding="10")
        self.agent_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.setup_metrics_display(self.agent_frame, "agent")

        # Comparison summary frame
        summary_frame = ttk.LabelFrame(self.root, text="COMPARISON SUMMARY", padding="10")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)

        self.setup_summary_display(summary_frame)

    def setup_metrics_display(self, parent, prefix):
        """Create labels to display metrics"""
        metrics = [
            ("Queue Length", "queue"),
            ("Waiting Time (s)", "waiting"),
            ("Reward", "reward"),
            ("Total Reward", "total_reward"),
            ("Sim Time (s)", "sim_time"),
            ("Phase Unchanged", "phase_unchanged")
        ]

        if not hasattr(self, 'metrics_labels') or self.metrics_labels is None:
            self.metrics_labels = {}

        for i, (label_text, key) in enumerate(metrics):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{label_text}:", width=20, anchor=tk.W).pack(side=tk.LEFT)

            value_label = ttk.Label(frame, text="0.00", width=15, anchor=tk.E, font=("Arial", 10, "bold"))
            value_label.pack(side=tk.LEFT, padx=5)

            self.metrics_labels[f"{prefix}_{key}"] = value_label

        # Status label
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=10)

        status_label = ttk.Label(status_frame, text="Status: Not Started", font=("Arial", 9))
        status_label.pack()
        self.metrics_labels[f"{prefix}_status"] = status_label

    def setup_summary_display(self, parent):
        """Create display for comparison summary"""
        summary_metrics = [
            ("Queue Length", "queue", "lower"),
            ("Waiting Time", "waiting", "lower"),
            ("Reward", "reward", "higher"),
            ("Sim Time", "sim_time", "lower")
        ]

        if not hasattr(self, 'summary_labels') or self.summary_labels is None:
            self.summary_labels = {}

        for label_text, key, better_direction in summary_metrics:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{label_text}:", width=20, anchor=tk.W).pack(side=tk.LEFT)

            baseline_label = ttk.Label(frame, text="0.00", width=12, anchor=tk.E)
            baseline_label.pack(side=tk.LEFT, padx=5)

            ttk.Label(frame, text="vs", width=5).pack(side=tk.LEFT)

            agent_label = ttk.Label(frame, text="0.00", width=12, anchor=tk.E)
            agent_label.pack(side=tk.LEFT, padx=5)

            result_label = ttk.Label(frame, text="---", width=20, anchor=tk.CENTER, font=("Arial", 9, "bold"))
            result_label.pack(side=tk.LEFT, padx=5)

            self.summary_labels[key] = {
                "baseline": baseline_label,
                "agent": agent_label,
                "result": result_label,
                "direction": better_direction
            }

    def select_model(self):
        """Select model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if filename:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, filename)
            self.model_path = filename

    def select_config(self):
        """Select config file"""
        filename = filedialog.askopenfilename(
            title="Select Config File",
            filetypes=[("SUMO config files", "*.sumocfg"), ("All files", "*.*")]
        )
        if filename:
            self.config_entry.delete(0, tk.END)
            self.config_entry.insert(0, filename)

    def select_route(self):
        """Select route file"""
        filename = filedialog.askopenfilename(
            title="Select Route File",
            filetypes=[("SUMO route files", "*.rou.xml"), ("XML files", "*.xml"), ("All files", "*.*")]
        )
        if filename:
            self.route_entry.delete(0, tk.END)
            self.route_entry.insert(0, filename)

    def reset_metrics(self):
        """Reset all metrics"""
        self.baseline_metrics.reset()
        self.agent_metrics.reset()
        self.update_display()

    def start_comparison(self):
        """Load and initialise both simulations (does not start them yet)"""
        if self.baseline_metrics.is_running or self.agent_metrics.is_running:
            messagebox.showwarning("Warning", "Simulation is running, please stop first!")
            return

        self.model_path = self.model_entry.get()
        config_file = self.config_entry.get()
        route_file = self.route_entry.get()

        if not self.model_path:
            messagebox.showerror("Error", "Please select a model file!")
            return

        # Reset metrics
        self.baseline_metrics.reset()
        self.agent_metrics.reset()
        self.baseline_port = None
        self.agent_port = None
        if self.baseline_frame:
            self.baseline_frame.config(text="BASELINE (Port: --)")
        if self.agent_frame:
            self.agent_frame.config(text="AGENT (Port: --)")
        self.stop_event.clear()
        self.start_event.clear()
        self.plot_triggered = False

        # Only initialize, not yet running
        self.baseline_metrics.is_running = False
        self.agent_metrics.is_running = False

        self.baseline_process = mp.Process(
            target=run_baseline_worker,
            args=(config_file, route_file, self.metrics_queue, self.stop_event, self.start_event),
            daemon=True
        )

        self.agent_process = mp.Process(
            target=run_agent_worker,
            args=(config_file, route_file, self.model_path, self.metrics_queue, self.stop_event, self.start_event),
            daemon=True
        )

        self.baseline_process.start()
        self.agent_process.start()

        time.sleep(2.0)

        self.start_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

    def run_simulations(self):
        """Start running both simulations"""
        if not self.baseline_process or not self.agent_process:
            messagebox.showwarning("Warning", "Please load simulation first!")
            return

        if not self.baseline_process.is_alive() or not self.agent_process.is_alive():
            messagebox.showwarning("Warning", "Simulation has been stopped, please reload!")
            return

        print("GUI: Setting start_event to begin both simulations...")
        self.start_event.set()
        print("GUI: start_event set")

        self.run_btn.config(state=tk.DISABLED)
        self.baseline_metrics.is_running = True
        self.agent_metrics.is_running = True

    def stop_comparison(self):
        """Stop comparison"""
        self.stop_event.set()
        self.baseline_metrics.is_running = False
        self.agent_metrics.is_running = False

        if self.baseline_process and self.baseline_process.is_alive():
            self.baseline_process.terminate()
            self.baseline_process.join(timeout=2)
        if self.agent_process and self.agent_process.is_alive():
            self.agent_process.terminate()
            self.agent_process.join(timeout=2)

        self.start_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.baseline_port = None
        self.agent_port = None
        if self.baseline_frame:
            self.baseline_frame.config(text="BASELINE (Port: --)")
        if self.agent_frame:
            self.agent_frame.config(text="AGENT (Port: --)")

    def get_comparison_color(self, baseline_val, agent_val, lower_is_better=True):
        """Return color for baseline and agent based on comparison"""
        if baseline_val == agent_val:
            return "black", "black"

        if lower_is_better:
            if baseline_val < agent_val:
                return "green", "red"
            else:
                return "red", "green"
        else:
            if baseline_val > agent_val:
                return "green", "red"
            else:
                return "red", "green"

    def update_display(self):
        """Update metrics display with comparison colors"""
        if not hasattr(self, 'metrics_labels') or not self.metrics_labels:
            return

        # Baseline
        if "baseline_queue" in self.metrics_labels:
            baseline_queue = self.baseline_metrics.current_queue
            agent_queue = self.agent_metrics.current_queue
            baseline_color, agent_color = self.get_comparison_color(baseline_queue, agent_queue, lower_is_better=True)
            self.metrics_labels["baseline_queue"].config(
                text=f"{baseline_queue:.2f}",
                foreground=baseline_color
            )

        if "baseline_waiting" in self.metrics_labels:
            baseline_waiting = self.baseline_metrics.current_waiting
            agent_waiting = self.agent_metrics.current_waiting
            baseline_color, agent_color = self.get_comparison_color(baseline_waiting, agent_waiting, lower_is_better=True)
            self.metrics_labels["baseline_waiting"].config(
                text=f"{baseline_waiting:.2f}",
                foreground=baseline_color
            )

        if "baseline_reward" in self.metrics_labels:
            baseline_reward = self.baseline_metrics.current_reward
            agent_reward = self.agent_metrics.current_reward
            baseline_color, agent_color = self.get_comparison_color(baseline_reward, agent_reward, lower_is_better=False)
            self.metrics_labels["baseline_reward"].config(
                text=f"{baseline_reward:.2f}",
                foreground=baseline_color
            )

        if "baseline_total_reward" in self.metrics_labels:
            baseline_total = self.baseline_metrics.total_reward
            agent_total = self.agent_metrics.total_reward
            baseline_color, agent_color = self.get_comparison_color(baseline_total, agent_total, lower_is_better=False)
            self.metrics_labels["baseline_total_reward"].config(
                text=f"{baseline_total:.2f}",
                foreground=baseline_color
            )

        if "baseline_sim_time" in self.metrics_labels:
            baseline_sim_time = self.baseline_metrics.current_sim_time
            agent_sim_time = self.agent_metrics.current_sim_time
            baseline_color, agent_color = self.get_comparison_color(baseline_sim_time, agent_sim_time, lower_is_better=True)
            self.metrics_labels["baseline_sim_time"].config(
                text=f"{baseline_sim_time:.2f}",
                foreground=baseline_color
            )

        if "baseline_phase_unchanged" in self.metrics_labels:
            phase_unchanged = self.baseline_metrics.current_phase_unchanged
            display_text = "True" if phase_unchanged else "False"
            color = "red" if phase_unchanged else "green"  # Red = useless green (bad), green = effective (good)
            self.metrics_labels["baseline_phase_unchanged"].config(
                text=display_text,
                foreground=color
            )

        if "baseline_status" in self.metrics_labels:
            if self.baseline_metrics.is_running:
                self.metrics_labels["baseline_status"].config(text="Status: Running...", foreground="blue")
            elif self.baseline_metrics.is_done:
                self.metrics_labels["baseline_status"].config(text="Status: Completed", foreground="green")
            elif self.baseline_process and self.baseline_process.is_alive():
                self.metrics_labels["baseline_status"].config(text="Status: Loaded (Stopped)", foreground="orange")
            else:
                self.metrics_labels["baseline_status"].config(text="Status: Not Started", foreground="black")

        # Agent
        if "agent_queue" in self.metrics_labels:
            baseline_queue = self.baseline_metrics.current_queue
            agent_queue = self.agent_metrics.current_queue
            baseline_color, agent_color = self.get_comparison_color(baseline_queue, agent_queue, lower_is_better=True)
            self.metrics_labels["agent_queue"].config(
                text=f"{agent_queue:.2f}",
                foreground=agent_color
            )

        if "agent_waiting" in self.metrics_labels:
            baseline_waiting = self.baseline_metrics.current_waiting
            agent_waiting = self.agent_metrics.current_waiting
            baseline_color, agent_color = self.get_comparison_color(baseline_waiting, agent_waiting, lower_is_better=True)
            self.metrics_labels["agent_waiting"].config(
                text=f"{agent_waiting:.2f}",
                foreground=agent_color
            )

        if "agent_reward" in self.metrics_labels:
            baseline_reward = self.baseline_metrics.current_reward
            agent_reward = self.agent_metrics.current_reward
            baseline_color, agent_color = self.get_comparison_color(baseline_reward, agent_reward, lower_is_better=False)
            self.metrics_labels["agent_reward"].config(
                text=f"{agent_reward:.2f}",
                foreground=agent_color
            )

        if "agent_total_reward" in self.metrics_labels:
            baseline_total = self.baseline_metrics.total_reward
            agent_total = self.agent_metrics.total_reward
            baseline_color, agent_color = self.get_comparison_color(baseline_total, agent_total, lower_is_better=False)
            self.metrics_labels["agent_total_reward"].config(
                text=f"{agent_total:.2f}",
                foreground=agent_color
            )

        if "agent_sim_time" in self.metrics_labels:
            baseline_sim_time = self.baseline_metrics.current_sim_time
            agent_sim_time = self.agent_metrics.current_sim_time
            baseline_color, agent_color = self.get_comparison_color(baseline_sim_time, agent_sim_time, lower_is_better=True)
            self.metrics_labels["agent_sim_time"].config(
                text=f"{agent_sim_time:.2f}",
                foreground=agent_color
            )

        if "agent_phase_unchanged" in self.metrics_labels:
            phase_unchanged = self.agent_metrics.current_phase_unchanged
            display_text = "True" if phase_unchanged else "False"
            color = "red" if phase_unchanged else "green"  # Red = useless green (bad), green = effective (good)
            self.metrics_labels["agent_phase_unchanged"].config(
                text=display_text,
                foreground=color
            )

        if "agent_status" in self.metrics_labels:
            if self.agent_metrics.is_running:
                self.metrics_labels["agent_status"].config(text="Status: Running...", foreground="blue")
            elif self.agent_metrics.is_done:
                self.metrics_labels["agent_status"].config(text="Status: Completed", foreground="green")
            elif self.agent_process and self.agent_process.is_alive():
                self.metrics_labels["agent_status"].config(text="Status: Loaded (Stopped)", foreground="orange")
            else:
                self.metrics_labels["agent_status"].config(text="Status: Not Started", foreground="black")

        self.update_summary()

    def update_summary(self):
        """Update comparison summary"""
        for key, labels in self.summary_labels.items():
            direction = labels["direction"]

            if key == "queue":
                baseline_val = self.baseline_metrics.current_queue
                agent_val = self.agent_metrics.current_queue
            elif key == "waiting":
                baseline_val = self.baseline_metrics.current_waiting
                agent_val = self.agent_metrics.current_waiting
            elif key == "reward":
                baseline_val = self.baseline_metrics.current_reward
                agent_val = self.agent_metrics.current_reward
            elif key == "sim_time":
                baseline_val = self.baseline_metrics.current_sim_time
                agent_val = self.agent_metrics.current_sim_time
            else:
                continue

            labels["baseline"].config(text=f"{baseline_val:.2f}")
            labels["agent"].config(text=f"{agent_val:.2f}")

            if baseline_val == 0 and agent_val == 0:
                result_text = "Equal"
                result_color = "black"
            elif direction == "lower":
                if agent_val < baseline_val:
                    result_text = "Agent BETTER"
                    result_color = "green"
                elif agent_val > baseline_val:
                    result_text = "Baseline BETTER"
                    result_color = "red"
                else:
                    result_text = "Equal"
                    result_color = "black"
            else:
                if agent_val > baseline_val:
                    result_text = "Agent BETTER"
                    result_color = "green"
                elif agent_val < baseline_val:
                    result_text = "Baseline BETTER"
                    result_color = "red"
                else:
                    result_text = "Equal"
                    result_color = "black"

            labels["result"].config(text=result_text, foreground=result_color)

    def start_update_loop(self):
        """Start GUI update loop"""
        try:
            while True:
                try:
                    msg = self.metrics_queue.get_nowait()
                    if msg[0] == "baseline_error" or msg[0] == "agent_error":
                        messagebox.showerror("Error", f"Error while running: {msg[1]}")
                        if msg[0] == "baseline_error":
                            self.baseline_metrics.is_running = False
                        else:
                            self.agent_metrics.is_running = False
                    elif msg[0] == "baseline_port":
                        self.baseline_port = msg[1]
                        if self.baseline_frame:
                            self.baseline_frame.config(text=f"BASELINE (Port: {self.baseline_port})")
                    elif msg[0] == "agent_port":
                        self.agent_port = msg[1]
                        if self.agent_frame:
                            self.agent_frame.config(text=f"AGENT (Port: {self.agent_port})")
                    elif msg[0] == "baseline_phase_duration":
                        phase_data = msg[1]
                        phase_id = phase_data["phase"]
                        duration = phase_data["duration"]
                        if phase_id in self.baseline_metrics.phase_durations:
                            self.baseline_metrics.phase_durations[phase_id].append(duration)
                    elif msg[0] == "baseline":
                        if msg[1] == "done":
                            self.baseline_metrics.is_running = False
                            self.baseline_metrics.is_done = True
                            if not self.baseline_metrics.episode_data:
                                self.baseline_metrics.episode_data = [{
                                    'queue': self.baseline_metrics.queue_lengths.copy(),
                                    'waiting': self.baseline_metrics.waiting_times.copy(),
                                    'reward': self.baseline_metrics.rewards.copy()
                                }]
                            if self.baseline_metrics.current_sim_time > 0:
                                self.baseline_metrics.sim_times.append(self.baseline_metrics.current_sim_time)
                        else:
                            metrics = msg[1]
                            self.baseline_metrics.current_queue = metrics["queue"]
                            self.baseline_metrics.current_waiting = metrics["waiting"]
                            self.baseline_metrics.current_reward = metrics["reward"]
                            self.baseline_metrics.current_sim_time = metrics["sim_time"]
                            self.baseline_metrics.current_phase_unchanged = metrics.get("phase_unchanged", False)
                            self.baseline_metrics.total_reward += metrics["reward"]
                            self.baseline_metrics.step_count += 1
                            self.baseline_metrics.queue_lengths.append(metrics["queue"])
                            self.baseline_metrics.waiting_times.append(metrics["waiting"])
                            self.baseline_metrics.rewards.append(metrics["reward"])
                    elif msg[0] == "agent_phase_duration":
                        phase_data = msg[1]
                        phase_id = phase_data["phase"]
                        duration = phase_data["duration"]
                        if phase_id in self.agent_metrics.phase_durations:
                            self.agent_metrics.phase_durations[phase_id].append(duration)
                    elif msg[0] == "agent":
                        if msg[1] == "done":
                            self.agent_metrics.is_running = False
                            self.agent_metrics.is_done = True
                            if not self.agent_metrics.episode_data:
                                self.agent_metrics.episode_data = [{
                                    'queue': self.agent_metrics.queue_lengths.copy(),
                                    'waiting': self.agent_metrics.waiting_times.copy(),
                                    'reward': self.agent_metrics.rewards.copy()
                                }]
                            if self.agent_metrics.current_sim_time > 0:
                                self.agent_metrics.sim_times.append(self.agent_metrics.current_sim_time)
                        else:
                            metrics = msg[1]
                            self.agent_metrics.current_queue = metrics["queue"]
                            self.agent_metrics.current_waiting = metrics["waiting"]
                            self.agent_metrics.current_reward = metrics["reward"]
                            self.agent_metrics.current_sim_time = metrics["sim_time"]
                            self.agent_metrics.current_phase_unchanged = metrics.get("phase_unchanged", False)
                            self.agent_metrics.total_reward += metrics["reward"]
                            self.agent_metrics.step_count += 1
                            self.agent_metrics.queue_lengths.append(metrics["queue"])
                            self.agent_metrics.waiting_times.append(metrics["waiting"])
                            self.agent_metrics.rewards.append(metrics["reward"])

                    if self.baseline_metrics.is_done and self.agent_metrics.is_done and not self.plot_triggered:
                        self.plot_triggered = True
                        self.root.after(500, self.plot_comparison)

                    self.update_display()
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Update loop error: {e}")

        self.root.after(100, self.start_update_loop)

    def plot_comparison(self):
        """Plot comparison charts between baseline and agent"""
        try:
            baseline_summary = self.baseline_metrics.get_summary()
            agent_summary = self.agent_metrics.get_summary()

            if not baseline_summary or not agent_summary:
                messagebox.showwarning("Warning", "Not enough data to plot!")
                return

            save_path = "result.png"

            fig = plt.figure(figsize=(18, 10))

            # 1. Queue comparison
            ax1 = plt.subplot(2, 3, 1)
            if 'queue_length' in baseline_summary and 'queue_length' in agent_summary:
                # Align series lengths by zero-padding the shorter one
                baseline_queue_raw = np.array(self.baseline_metrics.queue_lengths)
                agent_queue_raw = np.array(self.agent_metrics.queue_lengths)

                max_steps = max(len(baseline_queue_raw), len(agent_queue_raw))

                if len(baseline_queue_raw) < max_steps:
                    baseline_queue_padded = np.pad(baseline_queue_raw, (0, max_steps - len(baseline_queue_raw)), 'constant', constant_values=0)
                else:
                    baseline_queue_padded = baseline_queue_raw
                    
                if len(agent_queue_raw) < max_steps:
                    agent_queue_padded = np.pad(agent_queue_raw, (0, max_steps - len(agent_queue_raw)), 'constant', constant_values=0)
                else:
                    agent_queue_padded = agent_queue_raw

                # Mean over equal-length (zero-padded) series
                baseline_mean_filled = np.mean(baseline_queue_padded) if max_steps > 0 else 0
                agent_mean_filled = np.mean(agent_queue_padded) if max_steps > 0 else 0
                
                metrics = ['Mean Queue\n(All Steps)', 'Mean Queue\n(Filled to Max)', 'Max Queue']
                baseline_values = [
                    baseline_summary['queue_length']['mean'],
                    baseline_mean_filled,
                    baseline_summary['queue_length']['max'],
                ]
                agent_values = [
                    agent_summary['queue_length']['mean'],
                    agent_mean_filled,
                    agent_summary['queue_length']['max'],
                ]

                x = np.arange(len(metrics))
                width = 0.35

                bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='#FF6B6B', alpha=0.8)
                bars2 = ax1.bar(x + width/2, agent_values, width, label='Agent', color='#4ECDC4', alpha=0.8)

                ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Queue Length', fontsize=12, fontweight='bold')
                ax1.set_title('Figure 1: Queue Comparison', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(metrics, rotation=15, ha='right')
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3, axis='y')

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

            # 2. Waiting time comparison
            ax2 = plt.subplot(2, 3, 2)
            if 'waiting_time' in baseline_summary and 'waiting_time' in agent_summary:
                # Align series lengths by zero-padding the shorter one
                baseline_waiting_raw = np.array(self.baseline_metrics.waiting_times)
                agent_waiting_raw = np.array(self.agent_metrics.waiting_times)

                max_steps = max(len(baseline_waiting_raw), len(agent_waiting_raw))

                if len(baseline_waiting_raw) < max_steps:
                    baseline_waiting_padded = np.pad(baseline_waiting_raw, (0, max_steps - len(baseline_waiting_raw)), 'constant', constant_values=0)
                else:
                    baseline_waiting_padded = baseline_waiting_raw
                    
                if len(agent_waiting_raw) < max_steps:
                    agent_waiting_padded = np.pad(agent_waiting_raw, (0, max_steps - len(agent_waiting_raw)), 'constant', constant_values=0)
                else:
                    agent_waiting_padded = agent_waiting_raw

                # Mean over equal-length (zero-padded) series
                baseline_mean_filled = np.mean(baseline_waiting_padded) if max_steps > 0 else 0
                agent_mean_filled = np.mean(agent_waiting_padded) if max_steps > 0 else 0
                
                metrics = ['Mean Wait\n(All Steps)', 'Mean Wait\n(Filled to Max)', 'Max Wait']
                baseline_values = [
                    baseline_summary['waiting_time']['mean'],
                    baseline_mean_filled,
                    baseline_summary['waiting_time']['max'],
                ]
                agent_values = [
                    agent_summary['waiting_time']['mean'],
                    agent_mean_filled,
                    agent_summary['waiting_time']['max'],
                ]

                x = np.arange(len(metrics))
                width = 0.35

                bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', color='#FF6B6B', alpha=0.8)
                bars2 = ax2.bar(x + width/2, agent_values, width, label='Agent', color='#4ECDC4', alpha=0.8)

                ax2.set_xlabel('Metrics', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Waiting Time (s)', fontsize=12, fontweight='bold')
                ax2.set_title('Figure 2: Waiting Time Comparison', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(metrics, rotation=15, ha='right')
                ax2.legend(fontsize=11)
                ax2.grid(True, alpha=0.3, axis='y')

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

            # 3. Average Green Phase Duration by Phase (dual y-axis: left=mean, right=max)
            ax3 = plt.subplot(2, 3, 3)
            if 'phase_durations' in baseline_summary and 'phase_durations' in agent_summary:
                phases = [0, 2, 4, 6]
                phase_labels = ['Phase 0', 'Phase 2', 'Phase 4', 'Phase 6']

                baseline_means = []
                agent_means = []
                agent_maxs = []

                for phase_id in phases:
                    baseline_mean = baseline_summary['phase_durations'].get(phase_id, {}).get('mean', 0.0)
                    agent_mean = agent_summary['phase_durations'].get(phase_id, {}).get('mean', 0.0)
                    baseline_means.append(baseline_mean)
                    agent_means.append(agent_mean)

                    agent_durations = agent_summary['phase_durations'].get(phase_id, {}).get('all', [])
                    if agent_durations and len(agent_durations) > 0:
                        agent_maxs.append(float(np.max(agent_durations)))
                    else:
                        agent_maxs.append(0.0)

                x = np.arange(len(phase_labels))
                width = 0.22

                # Left axis: mean values (Baseline + Agent)
                bars1 = ax3.bar(x - width, baseline_means, width, label='Baseline Mean',
                                color='#FF6B6B', alpha=0.85)
                bars2 = ax3.bar(x,          agent_means,   width, label='Agent Mean',
                                color='#4ECDC4', alpha=0.85)

                ax3.set_xlabel('Phase', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Mean Green Duration (s)', fontsize=11, fontweight='bold', color='#333333')
                ax3.set_title('Figure 3: Green Phase Duration by Phase', fontsize=14, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(phase_labels)
                ax3.tick_params(axis='y', labelcolor='#333333')
                ax3.grid(True, alpha=0.3, axis='y')

                # Right axis: max values (Agent Max)
                ax3_right = ax3.twinx()
                bars3 = ax3_right.bar(x + width, agent_maxs, width, label='Agent Max',
                                      color='#E8A838', alpha=0.75, hatch='//')
                ax3_right.set_ylabel('Max Green Duration (s)', fontsize=11,
                                     fontweight='bold', color='#B87020')
                ax3_right.tick_params(axis='y', labelcolor='#B87020')

                # Value labels on mean bars (above bar, horizontal)
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)

                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{height:.1f}', ha='center', va='bottom', fontsize=8)

                # Value labels on max bars
                ax3_right_ylim = ax3_right.get_ylim()
                right_range = ax3_right_ylim[1] - ax3_right_ylim[0]
                for bar in bars3:
                    height = bar.get_height()
                    if height > 0:
                        bar_cx = bar.get_x() + bar.get_width() / 2.
                        if height >= 1000:
                            # Greater than 1000: rotate 90°, place in middle of bar to avoid overlap
                            if height > right_range * 0.15:
                                ax3_right.text(bar_cx, height * 0.5,
                                               f'{height:.1f}', ha='center', va='center',
                                               fontsize=8, color='#7A4A00', fontweight='bold',
                                               rotation=90)
                            else:
                                ax3_right.text(bar_cx, height,
                                               f'{height:.1f}', ha='center', va='bottom',
                                               fontsize=8, color='#B87020', fontweight='bold',
                                               rotation=90)
                        else:
                            # Below 1000: keep default (horizontal, on top of bar)

                            ax3_right.text(bar_cx, height,
                                           f'{height:.1f}', ha='center', va='bottom',
                                           fontsize=8, color='#B87020', fontweight='bold')

                # Combined legend from both axes
                handles1, labels1 = ax3.get_legend_handles_labels()
                handles2, labels2 = ax3_right.get_legend_handles_labels()
                ax3.legend(handles1 + handles2, labels1 + labels2, fontsize=9, loc='upper left')

            # 4. Queue length over steps
            ax4 = plt.subplot(2, 3, 4)
            if 'episode_data' in baseline_summary and 'episode_data' in agent_summary:
                baseline_episodes = baseline_summary['episode_data']
                agent_episodes = agent_summary['episode_data']

                # Concatenate only non-empty episode arrays to avoid numpy errors
                baseline_queue_arrays = [
                    np.array(ep['queue'], dtype=np.float32)
                    for ep in baseline_episodes
                    if len(ep['queue']) > 0
                ]
                if baseline_queue_arrays:
                    baseline_queue_series = np.concatenate(baseline_queue_arrays)
                else:
                    baseline_queue_series = np.array([], dtype=np.float32)

                agent_queue_arrays = [
                    np.array(ep['queue'], dtype=np.float32)
                    for ep in agent_episodes
                    if len(ep['queue']) > 0
                ]
                if agent_queue_arrays:
                    agent_queue_series = np.concatenate(agent_queue_arrays)
                else:
                    agent_queue_series = np.array([], dtype=np.float32)

                def smooth_series(series, window=20):
                    if series.size < window:
                        return np.arange(series.size), series
                    kernel = np.ones(window, dtype=np.float32) / window
                    smoothed = np.convolve(series, kernel, mode='valid')
                    x = np.arange(len(smoothed))
                    return x, smoothed

                if baseline_queue_series.size > 0:
                    t_baseline_raw = np.arange(len(baseline_queue_series))
                    ax4.plot(t_baseline_raw, baseline_queue_series, label='Baseline (raw)',
                            linewidth=0.5, color='#FF6B6B', alpha=0.2)
                    x_b, baseline_smooth = smooth_series(baseline_queue_series, window=20)
                    ax4.plot(x_b, baseline_smooth, label='Baseline (smooth)',
                            linewidth=2.0, color='#FF6B6B')

                if agent_queue_series.size > 0:
                    t_agent_raw = np.arange(len(agent_queue_series))
                    ax4.plot(t_agent_raw, agent_queue_series, label='Agent (raw)',
                            linewidth=0.5, color='#4ECDC4', alpha=0.2)
                    x_a, agent_smooth = smooth_series(agent_queue_series, window=20)
                    ax4.plot(x_a, agent_smooth, label='Agent (smooth)',
                            linewidth=2.0, color='#4ECDC4')

                all_vals = np.concatenate([baseline_queue_series, agent_queue_series]) \
                    if baseline_queue_series.size > 0 or agent_queue_series.size > 0 else np.array([])
                if all_vals.size > 0:
                    y_min = np.percentile(all_vals, 2)
                    y_max = np.percentile(all_vals, 98)
                    ax4.set_ylim(y_min, y_max)

                ax4.set_xlabel('Step', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Queue Length', fontsize=12, fontweight='bold')
                ax4.set_title('Figure 4: Queue Length over Steps', fontsize=14, fontweight='bold')
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3)

            # 5. Waiting time over steps
            ax5 = plt.subplot(2, 3, 5)
            if 'episode_data' in baseline_summary and 'episode_data' in agent_summary:
                baseline_episodes = baseline_summary['episode_data']
                agent_episodes = agent_summary['episode_data']

                # Concatenate only non-empty episode arrays to avoid numpy errors
                baseline_waiting_arrays = [
                    np.array(ep['waiting'], dtype=np.float32)
                    for ep in baseline_episodes
                    if len(ep['waiting']) > 0
                ]
                if baseline_waiting_arrays:
                    baseline_waiting_series = np.concatenate(baseline_waiting_arrays)
                else:
                    baseline_waiting_series = np.array([], dtype=np.float32)

                agent_waiting_arrays = [
                    np.array(ep['waiting'], dtype=np.float32)
                    for ep in agent_episodes
                    if len(ep['waiting']) > 0
                ]
                if agent_waiting_arrays:
                    agent_waiting_series = np.concatenate(agent_waiting_arrays)
                else:
                    agent_waiting_series = np.array([], dtype=np.float32)

                def smooth_series(series, window=20):
                    if series.size < window:
                        return np.arange(series.size), series
                    kernel = np.ones(window, dtype=np.float32) / window
                    smoothed = np.convolve(series, kernel, mode='valid')
                    x = np.arange(len(smoothed))
                    return x, smoothed

                if baseline_waiting_series.size > 0:
                    t_baseline_raw = np.arange(len(baseline_waiting_series))
                    ax5.plot(t_baseline_raw, baseline_waiting_series, label='Baseline (raw)',
                            linewidth=0.5, color='#FF6B6B', alpha=0.2)
                    x_b, baseline_smooth = smooth_series(baseline_waiting_series, window=20)
                    ax5.plot(x_b, baseline_smooth, label='Baseline (smooth)',
                            linewidth=2.0, color='#FF6B6B')

                if agent_waiting_series.size > 0:
                    t_agent_raw = np.arange(len(agent_waiting_series))
                    ax5.plot(t_agent_raw, agent_waiting_series, label='Agent (raw)',
                            linewidth=0.5, color='#4ECDC4', alpha=0.2)
                    x_a, agent_smooth = smooth_series(agent_waiting_series, window=20)
                    ax5.plot(x_a, agent_smooth, label='Agent (smooth)',
                            linewidth=2.0, color='#4ECDC4')

                all_vals = np.concatenate([baseline_waiting_series, agent_waiting_series]) \
                    if baseline_waiting_series.size > 0 or agent_waiting_series.size > 0 else np.array([])
                if all_vals.size > 0:
                    y_min = np.percentile(all_vals, 2)
                    y_max = np.percentile(all_vals, 98)
                    ax5.set_ylim(y_min, y_max)

                ax5.set_xlabel('Step', fontsize=12, fontweight='bold')
                ax5.set_ylabel('Waiting Time (s)', fontsize=12, fontweight='bold')
                ax5.set_title('Figure 5: Waiting Time over Steps', fontsize=14, fontweight='bold')
                ax5.legend(fontsize=9)
                ax5.grid(True, alpha=0.3)

            # 6. Reward over simulation time
            ax6 = plt.subplot(2, 3, 6)
            if 'episode_data' in baseline_summary and 'episode_data' in agent_summary:
                baseline_episodes = baseline_summary['episode_data']
                agent_episodes = agent_summary['episode_data']

                # Concatenate only non-empty episode arrays to avoid numpy errors
                baseline_reward_arrays = [
                    np.array(ep['reward'], dtype=np.float32)
                    for ep in baseline_episodes
                    if len(ep['reward']) > 0
                ]
                if baseline_reward_arrays:
                    baseline_rewards_series = np.concatenate(baseline_reward_arrays)
                else:
                    baseline_rewards_series = np.array([], dtype=np.float32)

                agent_reward_arrays = [
                    np.array(ep['reward'], dtype=np.float32)
                    for ep in agent_episodes
                    if len(ep['reward']) > 0
                ]
                if agent_reward_arrays:
                    agent_rewards_series = np.concatenate(agent_reward_arrays)
                else:
                    agent_rewards_series = np.array([], dtype=np.float32)

                def smooth_series(series, window=20):
                    if series.size < window:
                        return np.arange(series.size), series
                    kernel = np.ones(window, dtype=np.float32) / window
                    smoothed = np.convolve(series, kernel, mode='valid')
                    x = np.arange(len(smoothed))
                    return x, smoothed

                if baseline_rewards_series.size > 0:
                    t_baseline_raw = np.arange(len(baseline_rewards_series))
                    ax6.plot(t_baseline_raw, baseline_rewards_series, label='Baseline (raw)',
                            linewidth=0.5, color='#FF6B6B', alpha=0.2)
                    x_b, baseline_smooth = smooth_series(baseline_rewards_series, window=20)
                    ax6.plot(x_b, baseline_smooth, label='Baseline (smooth)',
                            linewidth=2.0, color='#FF6B6B')

                if agent_rewards_series.size > 0:
                    t_agent_raw = np.arange(len(agent_rewards_series))
                    ax6.plot(t_agent_raw, agent_rewards_series, label='Agent (raw)',
                            linewidth=0.5, color='#4ECDC4', alpha=0.2)
                    x_a, agent_smooth = smooth_series(agent_rewards_series, window=20)
                    ax6.plot(x_a, agent_smooth, label='Agent (smooth)',
                            linewidth=2.0, color='#4ECDC4')

                all_vals = np.concatenate([baseline_rewards_series, agent_rewards_series]) \
                    if baseline_rewards_series.size > 0 or agent_rewards_series.size > 0 else np.array([])
                if all_vals.size > 0:
                    y_min = np.percentile(all_vals, 2)
                    y_max = np.percentile(all_vals, 98)
                    ax6.set_ylim(y_min, y_max)

                ax6.set_xlabel('Step', fontsize=12, fontweight='bold')
                ax6.set_ylabel('Reward', fontsize=12, fontweight='bold')
                ax6.set_title('Figure 6: Reward over Steps', fontsize=14, fontweight='bold')
                ax6.legend(fontsize=9)
                ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

            messagebox.showinfo("Success", f"Chart plotted and saved at: {save_path}")

        except Exception as e:
            print(f"Error plotting chart: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Cannot plot chart: {e}")


def main():
    root = tk.Tk()
    app = ComparisonDemo(root)
    root.mainloop()


if __name__ == "__main__":
    # Multiprocessing guard for Windows
    mp.set_start_method('spawn', force=True)
    main()
