import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera_feed import CameraFeed
from ball_detection import BallDetector
from single_camera import SingleCameraEstimator
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor3D


def calculate_velocity(trajectory, time_step):
    """
    Calculates the velocity components (vx, vy, vz) based on the trajectory and time step.

    Args:
        trajectory (list): List of (x, y, z) positions.
        time_step (float): Time step between consecutive trajectory points.

    Returns:
        tuple: Velocity components (vx, vy, vz).
    """
    if len(trajectory) < 2:
        return (0, 0, 0)  # Not enough data to calculate velocity

    # Use the first two points to calculate velocity
    x1, y1, z1 = trajectory[0]
    x2, y2, z2 = trajectory[1]

    # Calculate velocity components
    vx = (x2 - x1) / time_step
    vy = (y2 - y1) / time_step
    vz = (z2 - z1) / time_step

    return vx, vy, vz


def calculate_floor_projection(initial_position, velocity):
    """
    Calculates the 2D floor projection of the trajectory based on the initial position and velocity.

    Args:
        initial_position (tuple): The (x, y, z) initial position of the object.
        velocity (tuple): The (vx, vy, vz) velocity components of the object.

    Returns:
        list: List of (x, y) positions representing the 2D projection on the floor.
    """
    x0, y0, z0 = initial_position
    vx, vy, vz = velocity
    g = 9.81  # Gravitational constant (m/s^2)

    # Calculate time to ground (y = 0 relative to camera height)
    relative_y0 = y0 - 1.0  # Adjust for the camera's height above the ground
    if relative_y0 <= 0:
        return []  # No motion if the object is already below the camera
    time_to_ground = np.sqrt(2 * relative_y0 / g)

    # Generate time steps for the motion
    t = np.linspace(0, time_to_ground, num=50)

    # Calculate the horizontal positions (x, z) over time
    floor_projection = [(x0 + vx * ti, z0 + vz * ti) for ti in t]

    return floor_projection


def visualize_3d_trajectory_with_projection(trajectory, initial_position, velocity, time_to_ground):
    """
    Visualizes the predicted 3D trajectory and its 2D floor projection.

    Args:
        trajectory (list): List of (x, y, z) positions.
        initial_position (tuple): The (x, y, z) initial position of the object.
        velocity (tuple): The (vx, vy, vz) velocity components of the object.
        time_to_ground (float): Estimated time for the ball to hit the ground in seconds.
    """
    if len(trajectory) < 2:
        print("Not enough trajectory data to plot.")
        return

    # Convert trajectory to a numpy array
    trajectory = np.array(trajectory)

    # Calculate 2D floor projection
    floor_projection = calculate_floor_projection(initial_position, velocity)

    # 3D Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Predicted 3D Trajectory", color='b')
    ax.scatter(initial_position[0], initial_position[1], initial_position[2], color='r', s=50, label="Initial Position")
    ax.text(initial_position[0], initial_position[1], initial_position[2],
            f"Time to Ground: {time_to_ground:.2f}s",
            fontsize=10, color='black')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title("Predicted 3D Trajectory")
    ax.legend()

    # 2D Floor Projection on the XY plane
    floor_projection = np.array(floor_projection)
    ax2 = fig.add_subplot(122)
    ax2.plot(floor_projection[:, 0], floor_projection[:, 1], label="2D Floor Projection", color='g')
    ax2.scatter(initial_position[0], initial_position[2], color='r', label="Initial Position")
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title("2D Floor Projection (XZ Plane)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Initialize components
    camera = CameraFeed(width=1200, height=720)
    detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
    single_camera = SingleCameraEstimator(known_object_width=0.0656, focal_length=700)  # Object size: 6.56 cm
    tracker = MotionTracker()
    trajectory_predictor = TrajectoryPredictor3D()

    # Runtime variables
    initial_position = None  # Store the initial position
    prediction_made = False  # Ensure only one prediction is made
    calibration_start_time = time.time()
    calibration_duration = 3.0  # Calibration duration in seconds

    try:
        while True:
            # Step 1: Capture frame
            frame = camera.get_frame()

            # Step 2: Detect ball
            center, bounding_box = detector.detect_ball(frame)

            # Calibration Phase: Show the initial position for 3 seconds
            if time.time() - calibration_start_time < calibration_duration:
                if center and bounding_box:
                    single_camera.draw_distance_on_frame(frame, 0, bounding_box)
                camera.show_frame(frame, window_name="Calibration Phase")
                continue

            # Step 3: Estimate distance (depth) after calibration
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

            # Step 5: Make a single trajectory prediction
            if not prediction_made and initial_position:
                trajectory = trajectory_predictor.predict_trajectory()

                if trajectory:
                    # Calculate velocity
                    velocity = calculate_velocity(trajectory, time_step=0.1)

                    # Time to ground based on current height
                    time_to_ground = np.sqrt(2 * (initial_position[1] - 1.0) / 9.81)  # Adjust for camera height

                    # Visualize trajectory
                    visualize_3d_trajectory_with_projection(trajectory, initial_position, velocity, time_to_ground)
                    prediction_made = True  # Ensure we only predict once

            # Step 6: Visualization (real-time frame updates)
            if center and bounding_box:
                single_camera.draw_distance_on_frame(frame, distance, bounding_box)
            camera.show_frame(frame, window_name="Ball Tracking and Distance Estimation")

            # Step 7: Exit on key press
            if camera.exit_requested():
                break

    except Exception as e:

        
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        camera.close_windows()


if __name__ == "__main__":
    main()
