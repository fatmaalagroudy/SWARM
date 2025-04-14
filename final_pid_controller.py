import airsimneurips as ans
import math
import time
import numpy as np

client = ans.MultirotorClient()

def setup_environment():
    """
    Establish connection and initialize the drone.
    """
    client.confirmConnection()
    print('Connection established')
    client.simLoadLevel('Soccer_Field_Easy')
    client.enableApiControl(vehicle_name="drone_1")
    client.arm(vehicle_name="drone_1")
    client.simStartRace(1)
    time.sleep(2)
    initial_position = ans.Vector3r(-4.25, -2.0, 3)
    initial_orientation = ans.Quaternionr(0, 0, 0, 4.71)
    drone_pose = ans.Pose(initial_position, initial_orientation)
    client.simSetVehiclePose(drone_pose, ignore_collison=True)
    client.simStartRace(1)

def fetch_gate_locations():
    """
    Retrieve gate positions and names from the scene.
    """
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
    """
    Check if the drone is within the specified radius of the gate.
    """
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
        """
        Compute the PID control output.
        """
        self.integral += error * self.time_step
        derivative = (error - self.previous_error) / self.time_step
        output = self.proportional_gain * error + self.integral_gain * self.integral + self.derivative_gain * derivative
        self.previous_error = error
        return output

class DroneDataLogger:
    """
    A class to log the drone's flight data at specified intervals.
    """
    def __init__(self, logging_interval=0.1, vehicle_name="drone_1"):
        self.logging_interval = logging_interval
        self.vehicle_name = vehicle_name
        self.last_log_time = time.time()
        self.logged_data = []

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
            self.last_log_time = current_time

    def get_logged_data(self):
        return self.logged_data
    
def compute_forward_vector(quaternion):
    """
    Compute the forward vector from the drone's orientation quaternion.
    """
    pitch = math.asin(max(-1.0, min(1.0, 2*(quaternion.w_val*quaternion.y_val - quaternion.z_val*quaternion.x_val))))
    yaw = math.atan2(2*(quaternion.w_val*quaternion.z_val + quaternion.x_val*quaternion.y_val), 1 - 2*(quaternion.y_val**2 + quaternion.z_val**2))
    fx = math.cos(pitch) * math.cos(yaw)
    fy = math.cos(pitch) * math.sin(yaw)
    fz = math.sin(pitch)
    return (fx, fy, fz)

def log_plot_reference(data_logger: DroneDataLogger):
    # These calls to simSetVehiclePose are only for logging reference states for plotting.
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
    pid_x = OneDimensionPID(proportional_gain=1.2, integral_gain=0.1, derivative_gain=0.1, time_step=time_step)
    pid_y = OneDimensionPID(proportional_gain=1.2, integral_gain=0.1, derivative_gain=0.1, time_step=time_step)
    pid_z = OneDimensionPID(proportional_gain=1.2, integral_gain=0.1, derivative_gain=0.1, time_step=time_step)
    
    data_logger = DroneDataLogger(logging_interval=0.1, vehicle_name="drone_1")

    for gate, target_position in gate_locations.items():
        print(f"Navigating to {gate} at position {target_position}")
        pid_x.integral = pid_y.integral = pid_z.integral = 0
        pid_x.previous_error = pid_y.previous_error = pid_z.previous_error = 0
        
        while not is_within_gate_sphere(target_position):
            state = client.getMultirotorState(vehicle_name="drone_1")
            current_position = state.kinematics_estimated.position
            current_velocity = state.kinematics_estimated.linear_velocity

            error_vector = np.array([
                target_position.x_val - current_position.x_val,
                target_position.y_val - current_position.y_val,
                target_position.z_val - current_position.z_val
            ])
            distance = np.linalg.norm(error_vector)
            if distance > 0:
                desired_direction = error_vector / distance
            else:
                desired_direction = np.array([0, 0, 0])
            
            base_speed = 8.0
            desired_velocity_vector = base_speed * desired_direction

            current_velocity_vector = np.array([
                current_velocity.x_val,
                current_velocity.y_val,
                current_velocity.z_val
            ])

            adjusted_desired_velocity = desired_velocity_vector

            velocity_error = adjusted_desired_velocity - current_velocity_vector
            
            control_x = pid_x.compute(velocity_error[0])
            control_y = pid_y.compute(velocity_error[1])
            control_z = pid_z.compute(velocity_error[2])

            command_velocity = current_velocity_vector + np.array([control_x, control_y, control_z])
            
            desired_yaw = math.atan2(desired_direction[1], desired_direction[0])
            desired_yaw_deg = math.degrees(desired_yaw)
            
            client.moveByVelocityAsync(command_velocity[0], command_velocity[1], command_velocity[2], time_step, vehicle_name="drone_1")
            
            data_logger.log_data(client)
            time.sleep(time_step)
    
    print("Race completed")
    time.sleep(1)
    
    log_plot_reference(data_logger)
    data_logger.plot_flight_path(gate_locations)

if __name__ == "__main__":
    execute_race()