import cv2

# Capture video from the first webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error opening camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is successfully read
    if not ret:
        print("Error reading frame")
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Wait for 'q' key to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
