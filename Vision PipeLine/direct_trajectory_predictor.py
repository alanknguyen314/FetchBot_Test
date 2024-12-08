import numpy as np

class TrajectoryPredictor3D:
    def __init__(self):
        self.positions = []  # List of (x, y, z) positions

    def update_positions(self, position):
        """
        Updates the list of tracked positions with the latest (x, y, z) coordinates.
        Limits the history to the last 10 positions.

        Args:
            position (tuple): The (x, y, z) position of the ball.
        """
        self.positions.append(position)
        if len(self.positions) > 10:
            self.positions.pop(0)

    def predict_trajectory(self, steps=50, gravity=-9.8, time_step=0.1):
        """
        Predicts the 3D trajectory of the object moving directly toward the camera.

        Args:
            steps (int): Number of future steps to predict.
            gravity (float): Gravitational acceleration affecting the z-axis.
            time_step (float): Time step for prediction in seconds.

        Returns:
            list: Predicted (x, y, z) positions.
        """
        if len(self.positions) < 2:
            return []  # Not enough data for prediction

        # Use the last position as the starting point
        x, y, z = self.positions[-1]

        # Calculate velocity components (assuming uniform motion in x, y)
        vx = 0  # No horizontal motion along x (ball is directly at the camera)
        vy = 0  # No horizontal motion along y
        vz = (self.positions[-1][2] - self.positions[-2][2]) / time_step

        trajectory = []
        for _ in range(steps):
            x += vx * time_step
            y += vy * time_step
            z += vz * time_step + 0.5 * gravity * time_step**2
            vz += gravity * time_step  # Update velocity due to gravity
            trajectory.append((x, y, z))
            if z <= 0:  # Stop when the ball hits the ground
                break

        return trajectory
