import numpy as np
import matplotlib.pyplot as plt
from Robot import Robot

def check_forward_kinematics():

    # Link lengths 2 and 1
    robot = Robot(np.array([[2], [2]]), np.array([[1], [1]]), np.array([[1], [1]]), 0)
    thetas = np.array([[np.pi/4], [np.pi/2]])  # Example joint angles
    
    frames = robot.fk(thetas)
    H_0_ee = frames[:,:,-1]
    x = H_0_ee[0,2]
    y = H_0_ee[1,2]
    th = np.arctan2(H_0_ee[1,0], H_0_ee[0,0])
    
    ee_position = robot.ee(thetas)

    for i in range(robot.dof):
        print(f"Frame {i}:")
        print(frames[:, :, i])
    print(f"End effector frame:")
    print(frames[:, :, -1])
    print(f"Manual computation: [{x}, {y}, {th}]")
    print(f"Robot computation: {ee_position}")

check_forward_kinematics()