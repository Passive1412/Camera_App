import numpy as np
import datetime
import time
import math
import sys
import cv2
import os

def main():
  cap = cv2.VideoCapture(0)
  frameRate = cap.get(5)
  x = 1
  start = time.time()
  while(cap.isOpened()):
    ret, frame = cap.read()
    timenow = datetime.datetime.now()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or (ret != True):
      break

    if timenow.minute % 5 == 0:
      filename = filename = '/home/pi/git/Camera_App/sample_images/image_' + str(timenow.day) + "_" +  str(timenow.hour) + "_" + str(timenow.minute) + ".jpg"
      if not os.path.exists(filename):
        try:
          cv2.imwrite(filename, frame)
        except:
          print("something failed")
        else:
          print(f"saved image {filename}")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
  print("done")
