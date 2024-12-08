import time
from camera_feed import CameraFeed
from ball_detection import BallDetector
from single_camera import SingleCameraEstimator
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_trajectory(trajectory, initial_position):
    """
    Visualizes the predicted 3D trajectory with the initial position.

    Args:
        trajectory (list): List of (x, y, z) positions.
        initial_position (tuple): The (x, y, z) initial position of the object.
    """
    if len(trajectory) < 2:
        return  # Not enough data to plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert trajectory to a numpy array
    trajectory = np.array(trajectory)

    # Plot the predicted 3D trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Predicted 3D Trajectory", color='b')

    # Plot the initial position
    ax.scatter(initial_position[0], initial_position[1], initial_position[2], color='r', s=50, label="Initial Position")

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title("Predicted 3D Trajectory with Initial Position")
    plt.legend()
    plt.show()

def main():
    # Initialize components
    camera = CameraFeed(width=1200, height=720)
    detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
    single_camera = SingleCameraEstimator(known_object_width=0.065, focal_length=700)
    tracker = MotionTracker()
    trajectory_predictor = TrajectoryPredictor3D()

    # Runtime variables
    trail_points = []
    initial_position = None  # Store the initial position
    prediction_made = False  # Ensure only one prediction is made
    prev_time = time.time()

    try:
        while True:
            # Step 1: Capture frame
            frame = camera.get_frame()

            # Step 2: Detect ball
            center, bounding_box = detector.detect_ball(frame)

            # Step 3: Estimate distance (depth)
            distance = None
            if bounding_box:
                bounding_box_width = bounding_box[2]
                distance = single_camera.estimate_distance(bounding_box_width)

            # Step 4: Update Kalman filter and trajectory predictor
            if center and distance is not None:
                tracker.correct(center)
                smoothed_center = tracker.predict()  # Smoothed 2D position
                trajectory_predictor.update_positions((*smoothed_center, distance))

                # Store the initial position if not already set
                if initial_position is None:
                    initial_position = (*smoothed_center, distance)

            # Step 5: Make a single trajectory prediction after 1-2 seconds
            current_time = time.time()
            if not prediction_made and current_time - prev_time > 2.0:  # After 2 seconds
                trajectory = trajectory_predictor.predict_trajectory()
                visualize_3d_trajectory(trajectory, initial_position)
                prediction_made = True  # Ensure we only predict once

            # Step 6: Visualization (real-time frame updates)
            if center and bounding_box:
                single_camera.draw_distance_on_frame(frame, distance, bounding_box)
            camera.show_frame(frame, window_name="Ball Tracking and Distance Estimation")

            # Step 7: Exit on key press
            if camera.exit_requested():
                break

            # Step 8: Limit trail points for visualization
            trail_points = trail_points[-50:]  # Keep the last 50 points for clarity

    except Exception as e:  
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        camera.close_windows()

if __name__ == "__main__":
    main()
