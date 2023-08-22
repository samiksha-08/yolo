import cv2
from ultralytics import YOLO
import torch

model = YOLO('best.pt')

resized_image= cv2.imread('frame2780.jpg')


results = model(resized_image)  # list of Results objects
print('Confidence')
result = results[0]
b = result.boxes[0]
print("Probability:", b.conf)
conf = b.conf[0].item()
print("Probability:", conf)


#Obtaining bounding boxes

tensor = torch.tensor(results[0].boxes.xyxy)
elements = tensor[0].tolist()
l = [round(element) for element in elements]
print(l)
#Drawing bounding boxe
image = cv2.rectangle(resized_image, (l[0], l[1]), (l[2],l[3]), (255,0,0), 2)

#Obtaining class name
a = results[0].boxes.cls

j = 0
c = []
s = ''
for i in a:
  c.append(int(i.item()))
  j += 1
if c[0]==0:
  s = 'Fresh cut onions'
if c[0] == 1:
  s = 'Semi cooked onions'
if c[0] == 2:
  s = 'Fully cooked onions'

#Adding text
text_position = (l[0], l[1] - 10)
cv2.putText(image, s, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
image = cv2.resize(resized_image, (640, 640))

#Displaying image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()