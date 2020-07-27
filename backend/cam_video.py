"""
A utility function to merge together many frames into a video.
"""
from progress.bar import Bar
import numpy as np
import copy
import glob
import time
import sys
import cv2
import os

DEBUG = True

def atoi(text):
    # A helper function to return digits inside text
    return int(text) if text.isdigit() else text

def natural_keys(text):
    # A helper function to generate keys for sorting frames AKA natural sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def make_heatmap():
  fileinput = '../output/project.avi'
  cap = cv2.VideoCapture(fileinput)
  fgbg = cv2.createBackgroundSubtractorMOG2()

  #num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  num_frames = 100

  bar = Bar('Processing Frames', max=num_frames)
  for i in range(0, num_frames):
    if i == 0:
      ret, frame = cap.read()
      first_frame = copy.deepcopy(frame)
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      height, width = gray.shape[:2]
      accum_image = np.zeros((height, width), np.uint8)
    else:
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      fgmask = fgbg.apply(gray)  
      cv2.imwrite('../output/diff-bkgnd-frame.jpg', fgmask)

      thresh = 2
      maxValue = 2
      ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
      #th2 = cv2.adaptiveThreshold(filter,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
      cv2.imwrite('diff-th1.jpg', th1)

      accum_image = cv2.add(accum_image, th1)
      cv2.imwrite('../output/mask.jpg', accum_image)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    bar.next()

  color_image = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
  cv2.imwrite('diff-color.jpg', color_image)

  result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
  cv2.imwrite('diff-overlay.jpg', result_overlay)

  bar.finish()
  cap.release()

  #make_video('/home/pi/git/Camera_App/output/frames/', '../output/heatmap.avi')    
  #name = "../output/frames/frame%d.jpg" % i
  #cv2.imwrite(name, video_frame)

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
  #make_video('/home/pi/git/Camera_App/sample_images/', '../output/project.avi')
  make_heatmap()
