import numpy as np

class Robot:
    """
    Represents a general fixed-base kinematic chain.
    """

    def __init__(self, link_lengths, link_masses, joint_masses, end_effector_mass):
        """
        Constructor: Makes a brand new robot with the specified parameters.
        """
        # Validate input
        if link_lengths.shape[1] != 1:
            raise ValueError(f"Invalid link_lengths: Should be a column vector, is {link_lengths.shape[0]}x{link_lengths.shape[1]}.")
        
        if link_masses.shape[1] != 1:
            raise ValueError(f"Invalid link_masses: Should be a column vector, is {link_masses.shape[0]}x{link_masses.shape[1]}.")
        
        if joint_masses.shape[1] != 1:
            raise ValueError("Invalid joint_masses: Should be a column vector.")
        
        if not isinstance(end_effector_mass, (int, float)):
            raise ValueError("Invalid end_effector_mass: Should be a number.")
        
        self.dof = link_lengths.shape[0]
        
        if link_masses.shape[0] != self.dof:
            raise ValueError("Invalid number of link masses: should match number of link lengths.")
        
        if joint_masses.shape[0] != self.dof:
            raise ValueError("Invalid number of joint masses: should match number of link lengths. Did you forget the base joint?")
        
        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.joint_masses = joint_masses
        self.end_effector_mass = end_effector_mass

    def forward_kinematics(self, thetas):
        """
        Returns the forward kinematic map for each frame, one for the base of
        each link, and one for the end effector. Link i is given by
        frames[:,:,i], and the end effector frame is frames[:,:,-1].
        """
        if thetas.shape[1] != 1:
            raise ValueError("Expecting a column vector of joint angles.")
        
        if thetas.shape[0] != self.dof:
            raise ValueError(f"Invalid number of joints: {thetas.shape[0]} found, expecting {self.dof}")
        
        # Allocate a variable containing the transforms from each frame
        # to the base frame.
        frames = np.zeros((3, 3, self.dof + 1))
        n = self.dof

        frames[:, :, 0] = np.eye(3) # Initialize the base frame as the identity matrix

        for i in range(n): # Iterate over each frame we need to translate between
            
            assert(i < n) # For good measure. i \in [0, n-1]

            # Rotation: Grab theta and create relevant trig
            theta = thetas[i][0] # Must decompose from 1-1 array to a scalar
            c = np.cos(theta)
            s = np.sin(theta)

            # Translation: Translate along positive x by link length
            tx = self.link_lengths[i][0] # Must decompose from 1-1 array to a scalar

            # Create homogeneous transform matrix for H^i_{i+1}
            this_H = np.array([ 
                [c, -s, tx * c], # idk apply the x transform in the rotated frame. I guess one could do
                [s,  c, tx * s], # a pure x translation first and then rotate the whole thing, but this
                [0,  0,      1]  # is not meaningfully more complex
                ])

            # Let the i+1th H be the product of the ith H and the current H
            base_frame = frames[:, :, i] 
            frames[:, :, i+1] = base_frame @ this_H
        
        return frames

    def fk(self, thetas):
        """
        Shorthand for returning the forward kinematics.
        """
        return self.forward_kinematics(thetas)

    def end_effector(self, thetas):
        """
        Returns [x; y; theta] for the end effector given a set of joint angles.
        """
        # Find the transform to the end-effector frame.
        frames = self.fk(thetas)
        H_0_ee = frames[:,:,-1]
        
        # Extract the components of the end_effector position and orientation.
        x = H_0_ee[0,2]
        y = H_0_ee[1,2]
        th = np.arctan2(H_0_ee[1,0], H_0_ee[0,0])
        
        # Pack them up nicely.
        return np.array([x, y, th])

    def ee(self, thetas):
        """
        Shorthand for returning the end effector position and orientation.
        """
        return self.end_effector(thetas)