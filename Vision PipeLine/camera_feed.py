import cv2

class CameraFeed:
    def __init__(self, width=1200, height=720):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera.")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to grab frame")
        return frame
    
    def release(self):
        self.cap.release()
    
    def show_frame(self, frame, window_name='Camera Feed'):
        cv2.imshow(window_name, frame)
    
    def exit_requested(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def close_windows(self):
        cv2.destroyAllWindows()
