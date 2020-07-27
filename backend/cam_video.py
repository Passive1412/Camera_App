"""

A utility function to merge together many frames into a video.

"""

from progress.bar import Bar
import numpy as np
import datetime
import copy
import glob
import time
import math
import sys
import cv2
import os

def atoi(text):
    # A helper function to return digits inside text
    return int(text) if text.isdigit() else text

def natural_keys(text):
    # A helper function to generate keys for sorting frames AKA natural sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_heatmap():
  fileinput = '../output/project.avi'
  capture = cv2.VideoCapture(fileinput)
  background_subtractor = cv2.createBackgroundSubtractorMOG2()
  length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

  bar = Bar('Processing Frames', max=length)
  for i in range(0, length):
    ret, frame = capture.read()
    #frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    if i == 0:
      first_frame = copy.deepcopy(frame)
      height, width, layers = frame.shape
      accum_image = np.zeros((height, width), np.uint8)

    else:
      filter = background_subtractor.apply(frame)  # remove the background
      cv2.imwrite('../output/frame.jpg', frame)
      cv2.imwrite('../output/diff-bkgnd-frame.jpg', filter)

      threshold = 127
      maxValue = 255
      ret, th0 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)
      th1 = cv2.adaptiveThreshold(filter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

      accum_image = cv2.add(accum_image, th1)
      cv2.imwrite('../output/mask.jpg', accum_image)

      color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_SUMMER)
      video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

      name = "../output/frames/frame%d.jpg" % i
      cv2.imwrite(name, video_frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    bar.next()

  bar.finish()

  make_video('/home/pi/git/Camera_App/output/frames/', '../output/heatmap.avi')

  color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
  result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

  cv2.imwrite('../output/diff-overlay.jpg', result_overlay)
  capture.release()

def make_video(input, output):
  img_array = sorted(glob.glob(input + '*.jpg'), key=os.path.getmtime)

  img = cv2.imread(img_array[0])
  height, width, layers = img.shape
  size = (width,height)

  out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
  bar = Bar('Creating Video', max=len(img_array))

  for img in img_array:
    out.write(cv2.imread(os.path.join(input, img)))
    bar.next()

  out.release()
  bar.finish()

if __name__ == "__main__":
  make_video('/home/pi/git/Camera_App/sample_images/', '../output/project.avi')
  #make_heatmap()
