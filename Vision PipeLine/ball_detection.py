import cv2
import numpy as np

class BallDetector:
    def __init__(self, lower_color_range, upper_color_range):
        """
        Initializes the ball detector with the specified HSV color range.

        Args:
            lower_color_range (list): Lower HSV bound for the object's color.
            upper_color_range (list): Upper HSV bound for the object's color.
        """
        self.lower_color = np.array(lower_color_range)
        self.upper_color = np.array(upper_color_range)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_ball(self, frame, debug=False):
        """
        Detects objects with the specified color range in the frame.

        Args:
            frame (numpy.ndarray): The input image frame.
            debug (bool): If True, visualizes the detection process.

        Returns:
            tuple: (center, bbox) where center is the coordinates of the detected object
                   and bbox is the bounding box of the object as (x, y, width, height).
                   Returns (None, None) if no object is detected.
        """
        # Downsample frame for performance
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        
        # Apply Gaussian blur and convert to HSV
        blurred_frame = cv2.GaussianBlur(small_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        
        # Adjust brightness in the color range
        brightness = np.mean(hsv[:, :, 2])
        adjust_factor = 10
        self.lower_color[2] = max(self.lower_color[2] - adjust_factor, 0)
        self.upper_color[2] = min(self.upper_color[2] + adjust_factor, 255)
        
        # Create mask for the specified color range
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # Morphological operations to reduce noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects = []
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter based on size
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Relaxed aspect ratio to include non-spherical objects
                    if 0.3 <= aspect_ratio <= 3.0:  # Allows elongated or wide shapes
                        center = (int((x + w // 2) / scale_factor), int((y + h // 2) / scale_factor))
                        detected_objects.append((center, (int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)), area))
        
        # Sort objects by size (largest first) and return the largest object
        detected_objects.sort(key=lambda obj: obj[2], reverse=True)
        if detected_objects:
            center, bbox, _ = detected_objects[0]
            
            if debug:
                # Draw bounding box and center on the frame
                cv2.rectangle(frame, bbox[:2], (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.imshow("Ball Detection Debug", mask)
            
            return center, bbox

        # No object detected
        return None, None
