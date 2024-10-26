import cv2
import numpy as np

def detect_laser(frame):
    """
    Detects the laser pointer in the given frame.
    
    Args:
        frame (numpy.ndarray): The current frame from the webcam.

    Returns:
        tuple: Coordinates of the laser pointer center and its radius.
    """
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting red laser pointers
    lower_red1 = np.array([0, 100, 100])  # Lower bound for red color
    upper_red1 = np.array([10, 255, 255])  # Upper bound for first red range
    lower_red2 = np.array([160, 100, 100])  # Lower bound for second red range
    upper_red2 = np.array([180, 255, 255])  # Upper bound for second red range
    
    # Create masks to isolate red colors in the frame
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2  # Combine the masks

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour, which should correspond to the laser pointer
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum enclosing circle around the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))  # Convert to integer coordinates
        
        # Draw the circle and center point on the frame
        cv2.circle(frame, center, int(radius), (0, 255, 0), 2)  # Draw enclosing circle
        cv2.circle(frame, center, 5, (255, 0, 0), -1)  # Draw center point
        return center, radius
    return None, None  # Return None if no laser detected

def main():
    """
    Main function to initialize the webcam and run the laser detection.
    """
    # Start video capture from the default camera (0)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect laser pointer in the current frame
        center, radius = detect_laser(frame)
        
        if center is not None:
            print(f"Laser detected at: {center}, Radius: {radius:.2f}")

        # Display the resulting frame
        cv2.imshow('Laser Pointer Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
