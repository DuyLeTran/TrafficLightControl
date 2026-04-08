import traci
import gymnasium as gym
from sumolib import checkBinary
import numpy as np
import os
class TrafficEnvironment(gym.Env):
    def __init__(self, config_file='SumoCfg/cross.sumocfg', step_length=0.1, gui=True, delay=0, route_file=None):
        super().__init__()
        self.config_file = config_file
        self.step_length = step_length
        self.gui = gui
        self.delay = delay
        self.route_file = route_file

        self.action_space = gym.spaces.Discrete(4)  # actions: 0..3 -> phases 0,2,4,6
        # observation = [queue_lengths(16) + vehicle_counts(16) + waiting_times(16)
        #                + current_phase_one_hot(8) + rt(1)] = 57 features
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(57,), dtype=np.float32)

        self.tls_id = "C"
        self.green_phases = [0, 2, 4, 6]
        self.current_phase = 0
        self.current_step = 0
        # Duration (seconds) that the current green phase has been active
        self.rt = 0.0
        # Previous queue_lengths and vehicle_counts for change detection
        self.prev_queue_lengths = None
        self.prev_vehicle_counts = None

        self.S1 = ['S1mp_to_C_0', 'S1_to_C_0',]
        self.S2 = ['S1mp_to_C_1', 'S1_to_C_1',]
        self.S3 = ['Smp2_to_C_0', 'Smp1_to_Smp2_0']
        self.S4 = ['Smp2_to_C_1', 'Smp1_to_Smp2_1']
        self.S5 = ['Smp2_to_C_2', 'Smp1_to_Smp2_2']
        self.S6 = ['Smp2_to_C_3', 'Smp1_to_Smp2_3']

        self.N1 = ['N1mp_to_C_0', 'N1_to_C_0',]  
        self.N2 = ['N1mp_to_C_1', 'N1_to_C_1',]
        self.N3 = ['Nmp2_to_C_0', 'Nmp1_to_Nmp2_0']
        self.N4 = ['Nmp2_to_C_1', 'Nmp1_to_Nmp2_1']
        self.N5 = ['Nmp2_to_C_2', 'Nmp1_to_Nmp2_2']
        self.N6 = ['Nmp2_to_C_3', 'Nmp1_to_Nmp2_3']
        self.W1 = ['Wmp_to_C_0', 'W_to_C_0']
        self.W2 = ['Wmp_to_C_1', 'W_to_C_1']

        self.E1 = ['Emp_to_C_0', 'E_to_C_0', 'E2mp_to_E_0', 'E2_to_E_0']
        self.E2 = ['Emp_to_C_1', 'E_to_C_1', 'E2mp_to_E_1', 'E2_to_E_1']

        self.lane_groups = {
            "S1": self.S1, "S2": self.S2, "S3": self.S3, "S4": self.S4, "S5": self.S5, "S6": self.S6,
            "N1": self.N1, "N2": self.N2, "N3": self.N3, "N4": self.N4, "N5": self.N5, "N6": self.N6,
            "W1": self.W1, "W2": self.W2,
            "E1": self.E1, "E2": self.E2,
        }

    def _init_sumo(self):
        binary = checkBinary('sumo-gui' if self.gui else 'sumo')
        args = [binary, "-c", self.config_file, "--start", "--quit-on-end"]
        # args = [binary, "-c", self.config_file, "--start"]
        # Suppress SUMO terminal output as much as possible
        args += [
            "--no-warnings",          # suppress warnings
            "--no-step-log",          # suppress per-step log
            "--no-duration-log",      # suppress performance/statistics summary
            "--verbose", "false",     # disable verbose mode
            "--print-options", "false",  # do not print options list
            "--waiting-time-memory", "10000", # increase waiting time memory window
        ]



        if self.step_length is not None:
            args += ["--step-length", str(self.step_length)]
        if self.delay is not None:
            args += ["--delay", str(self.delay)]
        if self.route_file is not None:
            args += ["--route-files", self.route_file]
        return args
    
    def _close_sumo(self):
        try:
            if traci.isLoaded():    
                traci.close()
        except traci.TraCIException:
            pass
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sumo()
        traci.start(self._init_sumo())
        self.current_step = 0
        self.current_phase = traci.trafficlight.getPhase(self.tls_id)
        # Reset green phase timer on environment reset
        self.rt = 0.0
        # Initialise previous-step values
        prev_queue = self._get_queue_length()
        prev_vehicle = self._get_vehicle_count()
        self.prev_queue_lengths = np.array(prev_queue, dtype=np.float32)
        self.prev_vehicle_counts = np.array(prev_vehicle, dtype=np.float32)
        observation = self._get_observation()
        return observation, {}
    
    def step(self, action):
        """
        Each environment step:
        - If the action requests a different green phase: run a yellow transition (not counted in rt),
          then switch to the new green phase and reset rt = 0.
        - Regardless of phase change, advance the simulation for one green-phase interval
          and add that duration to rt.
        """
        duration_green = 5  # green phase duration per decision step (seconds)

        # Snapshot queue_lengths and vehicle_counts before running the phase
        prev_queue = self._get_queue_length()
        prev_vehicle = self._get_vehicle_count()
        prev_queue_array = np.array(prev_queue, dtype=np.float32)
        prev_vehicle_array = np.array(prev_vehicle, dtype=np.float32)

        changed_phase = False
        if action*2 != self.current_phase:
            changed_phase = True
            # --- Yellow transition (5 s, not counted in rt) ---
            self.current_phase = (self.current_phase+1) % 8
            traci.trafficlight.setPhase(self.tls_id, self.current_phase)
            self._run_step(duration_green)
            # --- Switch to new green phase ---
            self.current_phase = action * 2
            # Reset green phase timer after switching
            self.rt = 0.0
        
        traci.trafficlight.setPhase(self.tls_id, self.current_phase)

        # Advance simulation in current green phase and accumulate rt
        self._run_step(duration_green)
        self.rt += duration_green

        observation = self._get_observation()

        # Raw (un-normalised) queue and vehicle counts after the phase step
        current_queue = np.array(self._get_queue_length(), dtype=np.float32)
        current_vehicle = np.array(self._get_vehicle_count(), dtype=np.float32)

        # Check whether the active green phase is ineffective (useless green)
        phase_unchanged = self._check_phase_unchanged(
            prev_queue_array, prev_vehicle_array, 
            current_queue, current_vehicle, 
            self.current_phase
        )

        reward = self._calculate_reward(observation, changed_phase, self.rt>120, phase_unchanged) # 120s is the maximum time of a green phase
        done = traci.simulation.getMinExpectedNumber() == 0
        info = (observation,phase_unchanged)
        # info = observation
        # Update previous-step values for the next decision step
        self.prev_queue_lengths = current_queue
        self.prev_vehicle_counts = current_vehicle
        
        observation = np.concatenate(observation, axis=0)  # flatten the observation for the agent
        return observation, reward, done, info



    # ===== Helpers =====
    def _run_step(self, duration=5):
        '''
        This function runs the SUMO simulation for a given duration (in seconds) by repeatedly calling traci.simulationStep().
        The main purpose is to prevent the agent from switching traffic light phases too rapidly, 
        ensuring a minimum time interval between decisions. It can also be used to simulate yellow (transitional) phase durations.
        '''
        for _ in range(int(duration / self.step_length)):
            traci.simulationStep()       

    def _get_queue_length(self):


        queue_lengths = {key: 0 for key in self.lane_groups.keys()}

        for key, lanes in self.lane_groups.items():
            for lane in lanes:
                queue_lengths[key] += traci.lane.getLastStepHaltingNumber(lane)

        return (
            queue_lengths["S1"], queue_lengths["S2"], queue_lengths["S3"], queue_lengths["S4"],
            queue_lengths["S5"], queue_lengths["S6"],
            queue_lengths["N1"], queue_lengths["N2"], queue_lengths["N3"], queue_lengths["N4"],
            queue_lengths["N5"], queue_lengths["N6"],
            queue_lengths["W1"], queue_lengths["W2"],
            queue_lengths["E1"], queue_lengths["E2"],
        )

    def _get_vehicle_count(self):
        """
        Returns total vehicle count (including moving vehicles) for each lane group
        S1..S6, N1..N6, W1..W2, E1..E2.
        """
        vehicle_counts = {key: 0 for key in self.lane_groups.keys()}

        for key, lanes in self.lane_groups.items():
            for lane in lanes:
                vehicle_counts[key] += traci.lane.getLastStepVehicleNumber(lane)

        return (
            vehicle_counts["S1"], vehicle_counts["S2"], vehicle_counts["S3"], vehicle_counts["S4"],
            vehicle_counts["S5"], vehicle_counts["S6"],
            vehicle_counts["N1"], vehicle_counts["N2"], vehicle_counts["N3"], vehicle_counts["N4"],
            vehicle_counts["N5"], vehicle_counts["N6"],
            vehicle_counts["W1"], vehicle_counts["W2"],
            vehicle_counts["E1"], vehicle_counts["E2"],
        )
    def _get_max_accumulated_waiting_time_per_lane(self):
        """
        Returns the maximum accumulated waiting time across all vehicles in each lane group.
        Iterates over all lanes in each group, collects vehicle IDs,
        and finds the highest accumulated waiting time among them.
        """
        accumulated_waiting_time = {key: 0.0 for key in self.lane_groups.keys()}
        for key, lanes in self.lane_groups.items():
            max_waiting_time = 0.0
            for lane in lanes:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                    if len(veh_ids) > 0:
                        for veh_id in veh_ids:
                            waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                            max_waiting_time = max(max_waiting_time, waiting_time)
                except Exception as e:
                    continue
            accumulated_waiting_time[key] = max_waiting_time
        return (
            accumulated_waiting_time["S1"], accumulated_waiting_time["S2"], accumulated_waiting_time["S3"], accumulated_waiting_time["S4"],
            accumulated_waiting_time["S5"], accumulated_waiting_time["S6"],
            accumulated_waiting_time["N1"], accumulated_waiting_time["N2"], accumulated_waiting_time["N3"], accumulated_waiting_time["N4"],
            accumulated_waiting_time["N5"], accumulated_waiting_time["N6"],
            accumulated_waiting_time["W1"], accumulated_waiting_time["W2"],
            accumulated_waiting_time["E1"], accumulated_waiting_time["E2"],
        )

    def _get_sum_accumulated_waiting_time_per_lane(self):
        """
        Returns the sum of accumulated waiting times across all vehicles in each lane group.
        Iterates over all lanes in each group, collects vehicle IDs,
        and sums their accumulated waiting times.
        """
        accumulated_waiting_time = {key: 0.0 for key in self.lane_groups.keys()}
        for key, lanes in self.lane_groups.items():
            sum_waiting_time = 0.0
            for lane in lanes:
                try:
                    veh_ids = traci.lane.getLastStepVehicleIDs(lane)
                    if len(veh_ids) > 0:
                        for veh_id in veh_ids:
                            waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                            sum_waiting_time += waiting_time
                except Exception as e:
                    continue
            accumulated_waiting_time[key] = sum_waiting_time
        return (
            accumulated_waiting_time["S1"], accumulated_waiting_time["S2"], accumulated_waiting_time["S3"], accumulated_waiting_time["S4"],
            accumulated_waiting_time["S5"], accumulated_waiting_time["S6"],
            accumulated_waiting_time["N1"], accumulated_waiting_time["N2"], accumulated_waiting_time["N3"], accumulated_waiting_time["N4"],
            accumulated_waiting_time["N5"], accumulated_waiting_time["N6"],
            accumulated_waiting_time["W1"], accumulated_waiting_time["W2"],
            accumulated_waiting_time["E1"], accumulated_waiting_time["E2"],
        )
    def _get_current_phase(self):
        one_hot_phases = np.zeros(8, dtype=np.float32)
        one_hot_phases[self.current_phase] = 1.0
        return one_hot_phases


    def _get_valid_actions(self):
        """
        Returns the list of valid actions — phases that have at least one vehicle present.
        Falls back to all actions if no vehicles exist anywhere (avoids deadlock).

        Action-to-lane-group index mapping:
          action 0 -> phase 0: S1(0), S4(3), S5(4), N1(6), N4(9),  N5(10)
          action 1 -> phase 2: S3(2), N3(8), W1(12), E1(14)
          action 2 -> phase 4: S2(1), S6(5), N2(7),  N6(11)
          action 3 -> phase 6: W2(13), E2(15)
        """
        vehicle_counts = np.array(self._get_vehicle_count(), dtype=np.float32)

        phase_to_indices = {
            0: [0, 3, 4, 6, 9, 10],
            1: [2, 8, 12, 14],
            2: [1, 5, 7, 11],
            3: [13, 15],
        }

        valid_actions = [
            action
            for action, indices in phase_to_indices.items()
            if any(vehicle_counts[i] > 0 for i in indices)
        ]

        # Fallback: if no vehicles exist anywhere, allow all actions
        return valid_actions if valid_actions else list(range(4))

    def _get_observation(self):
        queue_lengths = np.array(self._get_queue_length(), dtype=np.float32)
        vehicle_counts = np.array(self._get_vehicle_count(), dtype=np.float32)
        max_accumulated_waiting_time = np.array(self._get_max_accumulated_waiting_time_per_lane(), dtype=np.float32)
        current_phase = np.array(self._get_current_phase(), dtype=np.float32)
        rt = np.array([self.rt], dtype=np.float32)

        queue_lengths = queue_lengths / 80.0
        vehicle_counts = vehicle_counts / 80.0
        max_accumulated_waiting_time = max_accumulated_waiting_time / 120.0
        
        # Normalise rt to [0, 1] with max = 120 s (penalty threshold)
        rt = rt / 120.0

        return queue_lengths, vehicle_counts, max_accumulated_waiting_time, current_phase, rt

    def _check_phase_unchanged(self, prev_queue, prev_vehicle, current_queue, current_vehicle, current_phase):
        """
        Checks whether the active green phase is effective.
        Returns True if the phase is useless (will incur a heavy penalty):
        1. No vehicles in the phase at all (all vehicle counts = 0) — green with no cars to serve.
        2. Vehicles present but neither queue nor vehicle count changed — green that made no progress.

        Action-to-lane-group index mapping:
          action 0 -> phase 0: S1(0), S4(3), S5(4), N1(6), N4(9),  N5(10)
          action 1 -> phase 2: S3(2), N3(8), W1(12), E1(14)
          action 2 -> phase 4: S2(1), S6(5), N2(7),  N6(11)
          action 3 -> phase 6: W2(13), E2(15)
        """
        if current_phase not in self.green_phases:
            return False
        
        # Determine lane group indices for the active phase
        if current_phase == 0:  # phase_0 = [S1, S4, S5, N1, N4, N5]
            phase_indices = [0, 3, 4, 6, 9, 10]
        elif current_phase == 2:  # phase_2 = [S3, N3, W1, E1]
            phase_indices = [2, 8, 12, 14]
        elif current_phase == 4:  # phase_4 = [S2, S6, N2, N6]
            phase_indices = [1, 5, 7, 11]
        elif current_phase == 6:  # phase_6 = [W2, E2]
            phase_indices = [13, 15]
        else:
            return False

        # Check 1: all vehicle counts in the phase are zero — green with no cars to serve
        all_vehicle_zero = True
        for idx in phase_indices:
            if current_vehicle[idx] != 0:
                all_vehicle_zero = False
                break

        if all_vehicle_zero:
            return True

        # Check 2: penalise only if ALL lanes in the phase are ineffective.
        # A lane is considered effective if:
        #   - queue = 0 AND vehicle != 0 (vehicles are moving through), OR
        #   - queue > 0 AND (queue changed OR vehicle count changed)
        # A single effective lane is enough to skip the penalty.
        for idx in phase_indices:
            if current_queue[idx] == 0:
                # Vehicles moving (no queue but traffic present) — lane is effective
                if current_vehicle[idx] != 0:
                    return False
                # No vehicles at all on this lane — skip
                continue

            # queue > 0: check for progress
            queue_changed = prev_queue[idx] != current_queue[idx]
            vehicle_changed = prev_vehicle[idx] != current_vehicle[idx]

            if queue_changed or vehicle_changed:
                return False

        # All lanes in this phase are ineffective — apply penalty
        return True

    def _calculate_reward(self, observation, changed_phase: bool, rt: bool, phase_unchanged: bool):

        queue_lengths_normalized = observation[0]
        max_accumulated_waiting_time_per_lane_normalized = observation[2]
        
        max_accumulated_waiting_time_per_lane = np.array(self._get_max_accumulated_waiting_time_per_lane(), dtype=np.float32)
        
        R_linear = -0.25 * np.sum(queue_lengths_normalized) + -0.5*np.sum(max_accumulated_waiting_time_per_lane_normalized)
        R_progressive = -0.15* np.sum(np.maximum(0, max_accumulated_waiting_time_per_lane -120.0))

        reward = R_linear + R_progressive

        if changed_phase:
            reward += -5

        if rt:
            reward += -10
        
        if self.rt > 5 and phase_unchanged:
            reward += -15

        return reward
    
if __name__ == "__main__":
    env = TrafficEnvironment(route_file='SumoCfg/EWonly.rou.xml', delay=1000)
    env.reset()
    env.close()