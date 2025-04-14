
import airsimneurips as ans
import math
import time
import numpy as np

client = ans.MultirotorClient()

def setup_environment():
    client.confirmConnection()
    print('Connection established')
    
    # Load environment
    client.simLoadLevel("Soccer_Field_Easy")
    time.sleep(3)

    # Enable control and arm the drone
    client.enableApiControl(vehicle_name="drone_1")
    client.arm(vehicle_name="drone_1")
    client.takeoffAsync(vehicle_name="drone_1").join()

    # Set initial position right in front of the first gate (tweak as needed)
    initial_position = ans.Vector3r(-4.5, -2.0, -2.5)  # Z is negative = above ground
    initial_orientation = ans.Quaternionr(0, 0, 0, 1)
    drone_pose = ans.Pose(initial_position, initial_orientation)
    client.simSetVehiclePose(drone_pose, ignore_collison=True)

    time.sleep(1)  # Give physics engine a moment to stabilize

    # ✅ Optional: gentle push forward to start race cleanly
    client.moveByVelocityAsync(1.5, 0, -1.5, 1, vehicle_name="drone_1")
    time.sleep(1)

    # Start the race
    client.simStartRace(1)




def fetch_gate_locations():
    scene_objects = client.simListSceneObjects()
    gate_list = [obj for obj in scene_objects if 'Gate' in obj]
    gate_locations = {gate: client.simGetObjectPose(gate).position for gate in gate_list}

    def get_gate_index(gate_name):
        suffix = gate_name.replace("Gate", "")
        index_str = suffix.split("_")[0]
        try:
            return int(index_str)
        except ValueError:
            return float('inf')
    
    sorted_gate_locations = {gate: gate_locations[gate] for gate in sorted(gate_locations, key=get_gate_index)}
    print(sorted_gate_locations)
    return sorted_gate_locations

def is_within_gate_sphere(target_position: ans.Vector3r, radius=2):
    drone_state = client.getMultirotorState(vehicle_name="drone_1").kinematics_estimated.position
    delta_x = drone_state.x_val - target_position.x_val
    delta_y = drone_state.y_val - target_position.y_val
    delta_z = drone_state.z_val - target_position.z_val
    distance = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)
    
    if distance <= radius:
        print(f"Reached the sphere for gate at position {target_position}, moving to next gate")
        return True
    else:
        return False

class OneDimensionPID:
    def __init__(self, proportional_gain, integral_gain, derivative_gain, time_step):
        self.proportional_gain = proportional_gain
        self.integral_gain = integral_gain
        self.derivative_gain = derivative_gain
        self.time_step = time_step
        self.integral = 0
        self.previous_error = 0

    def compute(self, error):
        self.integral += error * self.time_step
        derivative = (error - self.previous_error) / self.time_step
        output = self.proportional_gain * error + self.integral_gain * self.integral + self.derivative_gain * derivative
        self.previous_error = error
        return output

class DroneDataLogger:
    def __init__(self, logging_interval=0.1, vehicle_name="drone_1"):
        self.logging_interval = logging_interval
        self.vehicle_name = vehicle_name
        self.last_log_time = time.time()
        self.logged_data = []
        self.velocity_data = []
        self.initial_gate_time = None
        self.final_gate_time = None
        self.planned_positions = []
        self.actual_positions = []

    def log_data(self, client):
        current_time = time.time()
        if current_time - self.last_log_time >= self.logging_interval:
            state = client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = (state.kinematics_estimated.position.x_val,
                   state.kinematics_estimated.position.y_val,
                   state.kinematics_estimated.position.z_val)
            q = state.kinematics_estimated.orientation
            ori = compute_forward_vector(q)
            vel = (state.kinematics_estimated.linear_velocity.x_val,
                   state.kinematics_estimated.linear_velocity.y_val,
                   state.kinematics_estimated.linear_velocity.z_val)
            self.logged_data.append({'pos': pos, 'ori': ori, 'vel': vel})
            self.velocity_data.append(vel)
            self.actual_positions.append(pos)
            self.last_log_time = current_time

    def get_logged_data(self):
        return self.logged_data

    def calculate_average_velocity(self):
        total_velocity = np.sum([np.linalg.norm(vel) for vel in self.velocity_data])
        return total_velocity / len(self.velocity_data) if len(self.velocity_data) > 0 else 0

    def calculate_navigation_ability(self):
        total_deviation = 0
        for planned, actual in zip(self.planned_positions, self.actual_positions):
            deviation = np.linalg.norm(np.array(planned) - np.array(actual))
            total_deviation += deviation
        return total_deviation / len(self.planned_positions) if len(self.planned_positions) > 0 else 0

    def set_initial_gate_time(self, time):
        self.initial_gate_time = time

    def set_final_gate_time(self, time):
        self.final_gate_time = time

    def calculate_time_to_completion(self):
        if self.initial_gate_time is not None and self.final_gate_time is not None:
            return self.final_gate_time - self.initial_gate_time
        else:
            return 0

def compute_forward_vector(quaternion):
    pitch = math.asin(max(-1.0, min(1.0, 2*(quaternion.w_val*quaternion.y_val - quaternion.z_val*quaternion.x_val))))
    yaw = math.atan2(2*(quaternion.w_val*quaternion.z_val + quaternion.x_val*quaternion.y_val), 1 - 2*(quaternion.y_val**2 + quaternion.z_val**2))
    fx = math.cos(pitch) * math.cos(yaw)
    fy = math.cos(pitch) * math.sin(yaw)
    fz = math.sin(pitch)
    return (fx, fy, fz)

def log_plot_reference(data_logger: DroneDataLogger):
    final_position = ans.Vector3r(25, 10, -20)
    final_orientation = ans.Quaternionr(0, 0, 0, 4.71)
    drone_pose = ans.Pose(final_position, final_orientation)
    client.simSetVehiclePose(drone_pose, ignore_collison=True)
    time.sleep(0.2)
    data_logger.log_data(client)
    final_position = ans.Vector3r(-3, -2.0, -20)
    final_orientation = ans.Quaternionr(0, 0, 0, 4.71)
    drone_pose = ans.Pose(final_position, final_orientation)
    client.simSetVehiclePose(drone_pose, ignore_collison=True)
    time.sleep(0.2)
    data_logger.log_data(client)


def execute_race():
    setup_environment()
    gate_locations = fetch_gate_locations()

    time_step = 0.01
    pid_x = OneDimensionPID(3, 0.001, 0.01, time_step)
    pid_y = OneDimensionPID(3, 0.001, 0.01, time_step)
    pid_z = OneDimensionPID(3, 0.001, 0.01, time_step)

    pos_pid_x = OneDimensionPID(1.0, 0.0, 0.0, time_step)
    pos_pid_y = OneDimensionPID(1.0, 0.0, 0.0, time_step)
    pos_pid_z = OneDimensionPID(1.0, 0.0, 0.0, time_step)

    data_logger = DroneDataLogger(logging_interval=0.1, vehicle_name="drone_1")

    gate_names = list(gate_locations.keys())
    for gate_name in gate_names:
        pos = gate_locations[gate_name]
        data_logger.planned_positions.append((pos.x_val, pos.y_val, pos.z_val))

    for i, gate in enumerate(gate_names):
        target_position = gate_locations[gate]

        print(f"Navigating to Gate {i+1}/{len(gate_names)} — {gate} at position {target_position}")

        for pid in [pid_x, pid_y, pid_z, pos_pid_x, pos_pid_y, pos_pid_z]:
            pid.integral = 0
            pid.previous_error = 0

        if i == 0:
            data_logger.set_initial_gate_time(time.time())

        last_gate = (i == len(gate_names) - 1)
        gate_radius = 3.0 if last_gate else 2.0

        start_time = time.time()
        while not is_within_gate_sphere(target_position, radius=gate_radius):
            if time.time() - start_time > 8:
                print(f"Timeout reached for {gate}, skipping to next.")
                break

            state = client.getMultirotorState(vehicle_name="drone_1")
            current_position = state.kinematics_estimated.position
            current_velocity = state.kinematics_estimated.linear_velocity

            pos_error = np.array([
                target_position.x_val - current_position.x_val,
                target_position.y_val - current_position.y_val,
                target_position.z_val - current_position.z_val
            ])

            hover_z_offset = -1.5
            desired_velocity = np.array([
                pos_pid_x.compute(pos_error[0]),
                pos_pid_y.compute(pos_error[1]),
                pos_pid_z.compute(pos_error[2]) + hover_z_offset
            ])

            current_velocity_vector = np.array([
                current_velocity.x_val,
                current_velocity.y_val,
                current_velocity.z_val
            ])

            velocity_error = desired_velocity - current_velocity_vector

            control_x = pid_x.compute(velocity_error[0])
            control_y = pid_y.compute(velocity_error[1])
            control_z = pid_z.compute(velocity_error[2])

            command_velocity = current_velocity_vector + np.array([control_x, control_y, control_z])

            desired_yaw = math.atan2(pos_error[1], pos_error[0])
            desired_yaw_deg = math.degrees(desired_yaw)

            client.moveByVelocityAsync(command_velocity[0], command_velocity[1], command_velocity[2], time_step, vehicle_name="drone_1")

            data_logger.log_data(client)
            time.sleep(time_step)

        if last_gate:
            data_logger.set_final_gate_time(time.time())

    print("Race completed")
    time.sleep(1)
    log_plot_reference(data_logger)

    time_to_completion = data_logger.calculate_time_to_completion()
    average_velocity = data_logger.calculate_average_velocity()
    navigation_ability = data_logger.calculate_navigation_ability()

    print(f"Time-to-Completion: {time_to_completion:.2f} seconds")
    print(f"Average Velocity: {average_velocity:.2f} m/s")
    print(f"Navigation Ability (Deviation Score): {navigation_ability:.2f} meters")


if __name__ == "__main__":
    execute_race()
