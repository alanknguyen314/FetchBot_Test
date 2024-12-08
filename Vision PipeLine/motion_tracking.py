import cv2
import numpy as np

class MotionTracker:
    def __init__(self):
        """
        Initializes the MotionTracker using a Kalman Filter for 2D tracking.
        """
        # Kalman filter setup
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # State and covariance initialization
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        """
        Predict the next position of the object.

        :return: Tuple (x, y) representing the predicted position in 2D space.
        """
        predicted = self.kalman.predict()
        return int(predicted[0]), int(predicted[1])

    def correct(self, measured_center):
        """
        Update the Kalman filter with the new measurement.

        :param measured_center: Tuple (x, y) of the object's detected position.
        :return: Tuple (x, y) representing the corrected position.
        """
        measurement = np.array([[np.float32(measured_center[0])], [np.float32(measured_center[1])]])
        corrected = self.kalman.correct(measurement)
        return int(corrected[0]), int(corrected[1])
