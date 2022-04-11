import cv2
import os
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_path = "/home/abhi/PycharmProjects/trabotyx/abhin_ws/scripts/HD2K_SN27744461_13-57-31.mp4"
cap = cv2.VideoCapture(video_path)

out_path = './lastyearsvo_3/'
if not os.path.isdir(out_path):
  os.makedirs(out_path, exist_ok=False)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  count += 1
  if ret == True:

    # Display the resulting frame
    # cv2.imshow('Frame', frame)
    # f_name = os.path.splitext(os.path.basename(video_path))[0]
    if count%15 == 0:
      cv2.imwrite(out_path+'lastyearsvo_3_img'+'_'+str(count)+'.jpg', frame)

    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #   break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()