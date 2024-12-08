import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera_feed import CameraFeed
from ball_detection import BallDetector
from single_camera import SingleCameraEstimator
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor3D


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

    # Calculate time to ground (z = 0)
    time_to_ground = np.sqrt(2 * z0 / g)

    # Generate time steps for the motion
    t = np.linspace(0, time_to_ground, num=50)

    # Calculate the horizontal positions (x, y) over time
    floor_projection = [(x0 + vx * ti, y0 + vy * ti) for ti in t]

    return floor_projection


def visualize_3d_trajectory_with_projection(trajectory, initial_position, velocity, time_to_ground):
    """
    Visualizes the predicted 3D trajectory and its 2D floor projection.

    Args:
        trajectory (list): List of (x, y, z) positions.
        initial_position (tuple): The (x, y, z) initial position of the object.
        velocity (float): Estimated velocity of the ball in m/s.
        time_to_ground (float): Estimated time for the ball to hit the ground in seconds.
    """
    if len(trajectory) < 2:
        print("Not enough trajectory data to plot.")
        return

    # Convert trajectory to a numpy array
    trajectory = np.array(trajectory)

    # Calculate 2D floor projection
    vx = velocity * (trajectory[1, 0] - trajectory[0, 0]) / np.linalg.norm(trajectory[1] - trajectory[0])
    vy = velocity * (trajectory[1, 1] - trajectory[0, 1]) / np.linalg.norm(trajectory[1] - trajectory[0])
    vz = velocity * (trajectory[1, 2] - trajectory[0, 2]) / np.linalg.norm(trajectory[1] - trajectory[0])

    floor_projection = calculate_floor_projection(initial_position, (vx, vy, vz))

    # 3D Plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Predicted 3D Trajectory", color='b')
    ax.scatter(initial_position[0], initial_position[1], initial_position[2], color='r', s=50, label="Initial Position")
    ax.text(initial_position[0], initial_position[1], initial_position[2],
            f"Velocity: {velocity:.2f} m/s\nTime to Ground: {time_to_ground:.2f}s",
            fontsize=10, color='black')
    ax.set_xlabel('X (centi-meters)')
    ax.set_ylabel('Y (centi-meters)')
    ax.set_zlabel('Z (centi-meters)')
    ax.set_title("Predicted 3D Trajectory")
    ax.legend()

    # 2D Floor Projection on the XY plane
    floor_projection = np.array(floor_projection)
    ax2 = fig.add_subplot(122)
    ax2.plot(floor_projection[:, 0], floor_projection[:, 1], label="2D Floor Projection", color='g')
    ax2.scatter(initial_position[0], initial_position[1], color='r', label="Initial Position")
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title("2D Floor Projection (XY Plane)")
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

                if trajectory:
                    # Use Kalman filter's smoothed 2D positions for displacement
                    x1, y1, z1 = trajectory[0]
                    x2, y2, z2 = trajectory[1]
                    displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                    time_step = current_time - prev_time  # Actual time step
                    velocity = displacement / time_step  # Velocity in m/s

                    # Adjust velocity for realistic scaling
                    velocity_scaled = velocity * 0.0656 / bounding_box_width

                    # Time to ground based on current height
                    time_to_ground = np.sqrt(2 * initial_position[2] / 9.81)  # Free-fall time calculation

                    # Visualize trajectory
                    visualize_3d_trajectory_with_projection(trajectory, initial_position, velocity_scaled, time_to_ground)
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
