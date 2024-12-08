import time
from camera_feed import CameraFeed
from ball_detection import BallDetector
from single_camera import SingleCameraEstimator
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_trajectory(trajectory):
    """
    Visualizes the predicted 3D trajectory.

    Args:
        trajectory (list): List of (x, y, z) positions.
    """
    if len(trajectory) < 2:
        return  # Not enough data to plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="3D Trajectory")
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title("Predicted 3D Trajectory")
    plt.legend()
    plt.show()

def main():
    # Initialize components
    camera = CameraFeed(width=1200, height=720)
    detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
    single_camera = SingleCameraEstimator(known_object_width=0.2, focal_length=700)
    tracker = MotionTracker()
    trajectory_predictor = TrajectoryPredictor3D()

    # Runtime variables
    trail_points = []
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
                trail_points.append((*smoothed_center, distance))

            # Step 5: Predict 3D trajectory
            trajectory = trajectory_predictor.predict_trajectory()

            # Step 6: Visualization (real-time frame updates)
            if center and bounding_box:
                single_camera.draw_distance_on_frame(frame, distance, bounding_box)
            camera.show_frame(frame, window_name="Ball Tracking and Distance Estimation")

            # Step 7: Display 3D trajectory visualization
            if len(trail_points) % 50 == 0:  # Visualize every 50 frames
                visualize_3d_trajectory(trajectory)

            # Step 8: Exit on key press
            if camera.exit_requested():
                break

            # Step 9: Limit trail points for visualization
            trail_points = trail_points[-50:]  # Keep the last 50 points for clarity

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        camera.close_windows()

if __name__ == "__main__":
    main()
