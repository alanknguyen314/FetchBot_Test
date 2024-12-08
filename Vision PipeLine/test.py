import time
from camera_feed import CameraFeed
from ball_detection import BallDetector
from motion_tracking import MotionTracker
from trajectory_prediction import TrajectoryPredictor
from visualization import Visualizer
from single_camera import SingleCameraEstimator

def main():
    # Initialize components
    camera = CameraFeed(width=1200, height=720)
    detector = BallDetector(lower_color_range=[40, 70, 70], upper_color_range=[80, 255, 255])
    tracker = MotionTracker()
    predictor = TrajectoryPredictor()
    visualizer = Visualizer()
    single_camera_estimator = SingleCameraEstimator(known_object_width=0.0658, focal_length=700)  # Example values

    # Initialize runtime variables
    trail_points = []
    predicted_trail_points = []
    prev_time = time.time()

    try:
        while True:
            # Step 1: Capture frame
            frame = camera.get_frame()

            # Step 2: Detect ball
            center, bounding_box = detector.detect_ball(frame)

            # Step 3: Estimate distance using single camera
            distance = None
            if bounding_box:
                bounding_box_width = bounding_box[2]  # Extract the width (w) of the bounding box
                distance = single_camera_estimator.estimate_distance(bounding_box_width)

            # Step 4: Update Kalman filter and get prediction
            predicted_center = tracker.predict()
            if center:
                tracker.correct(center)
                trail_points.append(center)
            predicted_trail_points.append(predicted_center)

            # Step 5: Update trajectory predictor
            if center:
                predictor.update_positions(center)

            predicted_landing = predictor.predict_landing(floor_y=720)  # Assuming the floor is at y=720

            # Step 6: Visualize results
            visualizer.draw_ball_info(frame, center, bounding_box)
            visualizer.draw_trajectory(frame, trail_points, color=(255, 0, 0))  # Blue for real trajectory
            visualizer.draw_trajectory(frame, predicted_trail_points, color=(0, 255, 255))  # Yellow for predicted trajectory
            if distance:
                single_camera_estimator.draw_distance_on_frame(frame, distance, bounding_box)

            # Step 7: Display metrics
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            mse = None
            accuracy = None
            if len(trail_points) > 5 and len(predicted_trail_points) > 5:
                real_points = trail_points[-5:]
                pred_points = predicted_trail_points[-5:]
                mse = sum(((r[0] - p[0])**2 + (r[1] - p[1])**2) for r, p in zip(real_points, pred_points)) / len(real_points)
                accuracy = sum(1 for r, p in zip(real_points, pred_points) if abs(r[0] - p[0]) <= 10 and abs(r[1] - p[1]) <= 10) / len(real_points) * 100

            visualizer.display_metrics(frame, fps, mse, accuracy)

            # Step 8: Show frame
            camera.show_frame(frame, window_name="Ball Tracking and Distance Estimation")

            # Step 9: Exit on key press
            if camera.exit_requested():
                break

            # Maintain trail point size
            trail_points = trail_points[-50:]  # Keep the last 50 points for visual clarity
            predicted_trail_points = predicted_trail_points[-50:]

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        camera.close_windows()

if __name__ == "__main__":
    main()
