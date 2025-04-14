import airsimneurips as ans
import time

client = ans.MultirotorClient()
client.confirmConnection()

print('Connection established')

# Load environment
client.simLoadLevel("Soccer_Field_Easy")
time.sleep(3)  # wait a bit for environment to fully load

# Control & arm
client.enableApiControl(vehicle_name="drone_1")
client.arm(vehicle_name="drone_1")

client.simStartRace(1)

# Takeoff
client.takeoffAsync(vehicle_name="drone_1").join()

# Just lift upward at 2 m/s for 3 seconds
print("Lifting up...")
client.moveByVelocityAsync(0, 0, -2, 3, vehicle_name="drone_1")
time.sleep(3)

# Hover in place
client.moveByVelocityAsync(0, 0, 0, 2, vehicle_name="drone_1")
time.sleep(2)

print("Test complete.")
