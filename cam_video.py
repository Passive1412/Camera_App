import numpy as np
import datetime
import glob
import time
import math
import sys
import cv2
import os


def main():
  filepath = '/home/pi/git/Camera_App/sample_images/'
  filelist = sorted(glob.glob(filepath + '*.jpg'), key=os.path.getmtime)
  img_array = []
  for filename in filelist:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

  out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()

if __name__ == "__main__":
  main()
