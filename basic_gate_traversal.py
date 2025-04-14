import airsimneurips

# 1. Connect to the AirSim simulation server with drone controls.
# This establishes a connection between our Python script and the simulation environment.
client = airsimneurips.MultirotorClient()
client.confirmConnection()
print('Connection confirmed')  # Confirms that connection to the simulation is successful

"""
Important:
- First, run the AirSim executable to load the simulation level.
- Uncomment `client.simLoadLevel('Soccer_Field_Easy')` to load a level.
- After loading, comment it out again and run the script below to avoid reloading every time.
"""

# Uncomment the line below to load the "Soccer_Field_Easy" level
client.simLoadLevel('Soccer_Field_Easy')

# 2. List all objects in the simulation and filter only the gates.
# Retrieve a list of objects in the scene and filter out those with the name 'Gate'.
objects = client.simListSceneObjects()
gates = [obj for obj in objects if 'Gate' in obj]  # We only want objects that contain 'Gate' in their names

# 3. Get the position of each gate and store the results in a dictionary.
# The gate's name is the key, and its position is the value.
gate_positions = {gate: client.simGetObjectPose(gate).position for gate in gates}
print(gate_positions)  # Prints the dictionary with the gate names and their respective positions

# 4. (Example: Optional) Fly the drone to an arbitrary position (x, y, z).
# Uncomment this section when you're ready to move the drone to this position.
client.enableApiControl(vehicle_name="drone_1")  # Enable API control for the drone
client.arm(vehicle_name="drone_1")  # Arm the drone so it can take off 
client.simStartRace()
client.takeoffAsync(vehicle_name="drone_1").join()

# Example of flying to a fixed arbitrary position, e.g., (x=10, y=10, z=-5).
# here the arguments for moveToPositionAsync are (x, y, z, velocity, vehicle_name)
#client.moveToPositionAsync(10, 10, -5, 5, vehicle_name="drone_1").join()

# 5. Fly the drone to each gate's position in a loop.
# The drone will fly to the x, y, z coordinates of each gate.

for gate, position in gate_positions.items():
    print(f"Flying to {gate} at position {position}")
    client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, 5, vehicle_name="drone_1").join()

# Note:
# - `moveToPositionAsync(x, y, z, velocity)` moves the drone to the specified coordinates (x, y, z) at a certain velocity - try changing the velocity to see how it affects the race time
# - `join()` ensures the program waits until the drone reaches the target before moving to the next gate.
