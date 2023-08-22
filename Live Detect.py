""" import cv2
from ultralytics import YOLO
import torch
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Define source as YouTube video URL
source = 'sample2.mp4'

# Run inference on the source
results = model(source, show=True)  # generator of Results objects
print(results)
 """

import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video fileq
video_path = "topview.mp4"
#video_path=2 #for accesing external webcam

#video_path = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)


# Set the resolution of the video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
print("FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))
count = 0
frame_count = 25  
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:

        if (count%frame_count == 0):
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            resized_image = cv2.resize(frame, (300, 300))

            results = model(resized_image)
            print("Inference Time: {}".format(time.time()-start_time))
            # print('Results')
            # print(results)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # print('Frame')
            # print(annotated_frame)
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break
    count += 1

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

""" 

import cv2
from ultralytics import YOLO
import torch

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Read an image using OpenCV
image = cv2.imread('frame2720.jpg')
source = cv2.resize(image, (640, 640))


# Run inference on the source
results = model(source)  # list of Results objects

#print("Bounding Boxes :",results[0].boxes.xyxy)

frame = results[0].plot()
cv2.imshow("YOLOv8 Inference", frame)

tensor = torch.tensor(results[0].boxes.xyxy)

# Extract each argument to a list
elements = tensor[0].tolist()
l = [round(element) for element in elements]


image = cv2.rectangle(source, (l[0], l[1]), (l[2],l[3]), (255,0,0), 2)
resized_image = cv2.resize(image, (640, 640))

cv2.imshow('Image', resized_image) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""