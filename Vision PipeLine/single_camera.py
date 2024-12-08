import cv2
import numpy as np
from collections import deque

class SingleCameraEstimator:
    def __init__(self, known_object_width, focal_length, max_history=10):
        """
        Initializes the SingleCameraEstimator.

        :param known_object_width: The real-world width of the object in meters.
        :param focal_length: The camera's focal length in pixels (calculated via calibration).
        :param max_history: Maximum number of bounding box widths to store for smoothing.
        """
        self.known_object_width = known_object_width
        self.focal_length = focal_length
        self.width_history = deque(maxlen=max_history)  # Store the last few bounding box widths for averaging

    def calibrate_focal_length(self, known_distance, bounding_box_width):
        """
        Calibrates the camera's focal length based on a known distance and bounding box width.

        :param known_distance: The real-world distance to the object in meters.
        :param bounding_box_width: Width of the object's bounding box in pixels.
        """
        if known_distance <= 0 or bounding_box_width <= 0:
            raise ValueError("Known distance and bounding box width must be positive.")
        self.focal_length = (bounding_box_width * known_distance) / self.known_object_width

    def estimate_distance(self, bounding_box_width):
        """
        Estimates the distance to the object based on its bounding box width in pixels.

        :param bounding_box_width: Width of the object's bounding box in pixels.
        :return: Estimated distance in meters.
        """
        if bounding_box_width <= 0:
            raise ValueError("Bounding box width must be positive.")
        
        # Smooth bounding box widths using a rolling average
        self.width_history.append(bounding_box_width)
        smoothed_width = np.mean(self.width_history)

        # Estimate distance using the pinhole camera model
        distance = (self.known_object_width * self.focal_length) / smoothed_width
        return distance

    def draw_distance_on_frame(self, frame, distance, bounding_box):
        """
        Annotates the frame with the estimated distance and bounding box.

        :param frame: The video frame.
        :param distance: The estimated distance in meters.
        :param bounding_box: The object's bounding box (x, y, w, h).
        """
        if bounding_box is None or len(bounding_box) != 4:
            return  # Do nothing if bounding box is invalid

        x, y, w, h = bounding_box
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Annotate the distance
        cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def handle_lost_tracking(self, frame, bounding_box):
        """
        Handles cases where tracking is lost, and provides a visual cue.
        """
        if bounding_box is None:
            cv2.putText(frame, "Object Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
