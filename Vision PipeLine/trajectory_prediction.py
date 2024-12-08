import numpy as np
from sklearn.linear_model import LinearRegression

class TrajectoryPredictor3D:
    def __init__(self):
        self.positions = []  # List of (x, y, z)

    def update_positions(self, position):
        self.positions.append(position)
        if len(self.positions) > 10:  # Limit to last 10 positions
            self.positions.pop(0)

    def predict_trajectory(self, steps=50, gravity=-9.8, time_step=0.1):
        """
        Predicts the 3D trajectory of the object.
        
        Args:
            steps (int): Number of future steps to predict.
            gravity (float): Gravitational acceleration affecting the z-axis.
            time_step (float): Time step for prediction in seconds.
        
        Returns:
            list: Predicted (x, y, z) positions.
        """
        if len(self.positions) < 3:
            return []

        trajectory = []
        x, y, z = self.positions[-1]
        vx = (self.positions[-1][0] - self.positions[-2][0]) / time_step
        vy = (self.positions[-1][1] - self.positions[-2][1]) / time_step
        vz = (self.positions[-1][2] - self.positions[-2][2]) / time_step

        for _ in range(steps):
            x += vx * time_step
            y += vy * time_step
            z += vz * time_step + 0.5 * gravity * time_step**2
            vz += gravity * time_step
            trajectory.append((x, y, z))

        return trajectory
