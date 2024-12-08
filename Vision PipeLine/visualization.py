import cv2

class Visualizer:
    def draw_ball_info(self, frame, center, bounding_box):
        if center and bounding_box:
            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {center}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def draw_trajectory(self, frame, points, color):
        if len(points) > 1:
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 2)
    
    def display_metrics(self, frame, fps, mse, accuracy):
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if mse is not None:
            cv2.putText(frame, f"MSE: {mse:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if accuracy is not None:
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
