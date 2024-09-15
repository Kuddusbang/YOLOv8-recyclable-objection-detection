import cv2
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO("/Users/towfiislam/Desktop/COMP4301/ProjectF/runs/detect/train12/weights/best.pt")

# Open the default camera (0 for the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Make detections on the frame
    results = model(frame)

    # Create a copy of the frame for annotation
    annotated_frame = frame.copy()

    # Iterate through the detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            # Get the class name from the model
            cls = int(box.cls)
            class_name = model.names[cls]

            # Draw the bounding box and label on the annotated frame
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "recyclable", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the annotated frame
    cv2.imshow("Live Object Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()