"""
A utility function to merge together many frames into a video.
"""
from datetime import datetime
from progress.bar import Bar
import numpy as np
import copy
import glob
import time
import sys
import cv2
import os

DEBUG = True

def drawLabel(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

def drawTitle(frame, text):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    try:
      h, w, layers = frame.shape
      pos = (int(w/2 - txt_size[0][0]/2), txt_size[0][1] + margin)
      end_x = pos[0] + txt_size[0][0] + margin
      end_y = pos[1] - txt_size[0][1] - margin

      cv2.rectangle(frame, pos, (end_x, end_y), (255,255,255), thickness)
      cv2.putText(frame, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    except:
      print("test") 

def drawTime(frame, file):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    timestamp = os.path.getmtime(file)
    text = str(datetime.fromtimestamp(timestamp)).split(".")[0]
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    try:
      h, w, layers = frame.shape
      pos = (int(w/2 - txt_size[0][0]/2), h - 3*txt_size[0][1] + margin)
      end_x = pos[0] + txt_size[0][0] + margin
      end_y = pos[1] - txt_size[0][1] - margin
      cv2.rectangle(frame, pos, (end_x, end_y), (255,255,255), thickness)
      cv2.putText(frame, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    except:
      print(file)


def atoi(text):
    # A helper function to return digits inside text
    return int(text) if text.isdigit() else text

def natural_keys(text):
    # A helper function to generate keys for sorting frames AKA natural sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def make_heatmap():
  fileinput = '../output/project_2.avi'
  cap = cv2.VideoCapture(fileinput)
  fgbg = cv2.createBackgroundSubtractorMOG2()
  #fgbg = cv2.createBackgroundSubtractorKNN()

  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  #num_frames = 150

  bar = Bar('Processing Frames', max=num_frames)
  for i in range(0, num_frames):
    ret, frame = cap.read()
    h, w, layers = frame.shape
    a = 125
    frame = frame[int(h/2-a):int(h/2+a), int(w/2-a):int(w/2+a)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.inRange(gray, 50, 175)
    cv2.imwrite('../output/gray.jpg', gray)
    if i == 0:
      first_frame = copy.deepcopy(frame)
      height, width = gray.shape[:2]
      accum_image = np.zeros((height, width), np.uint8)
    else:
      fgmask = fgbg.apply(gray)
      cv2.imwrite('../output/diff-bkgnd-frame.jpg', fgmask)

      thresh = 4
      maxValue = 4
      ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)
      #th1 = cv2.adaptiveThreshold(fgmask,2,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
      #th1 = cv2.adaptiveThreshold(fgmask,2,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
      cv2.imwrite('../output/diff-th1.jpg', th1)

      accum_image = cv2.add(accum_image, th1)
      cv2.imwrite('../output/mask.jpg', accum_image)

      color_image_video = im_color = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
      video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

      name = "../output/frames/frame%d.jpg" % i
      cv2.imwrite(name, video_frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    bar.next()

  cv2.imwrite('../output/diff-color.jpg', color_image_video)

  result_overlay = cv2.addWeighted(first_frame, 0.7, color_image_video, 0.7, 0)
  cv2.imwrite('../output/diff-overlay.jpg', result_overlay)

  bar.finish()
  cap.release()

  make_video('/home/pi/git/Camera_App/output/frames/', '../output/heatmap.avi')    

def filter():
  pass

def make_video(input, output, frame_count=None):
  img_array = sorted(glob.glob(input + '*.jpg'), key=os.path.getmtime)
  img_array.extend(sorted(glob.glob('/home/pi/git/Camera_App/sample_images_2/' + '*.jpg'), key=os.path.getmtime))

  img = cv2.imread(img_array[0])
  height, width, layers = img.shape
  size = (width,height)

  out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'DIVX'), 120, size)

  bar = Bar('Creating Video', max=len(img_array))
  for idx, img in enumerate(img_array):
    img_path = os.path.join(input, img)
    timestamp = os.path.getmtime(img_path)
    time_date = datetime.fromtimestamp(timestamp)

    if str(time_date.minute)[-1] == "0":
      bar.next()
      continue

    if str(time_date.day) == "3":
      if any(x in str(time_date.hour) for x in ["12","13","14","15","16","17"]):
        continue

    img_path = os.path.join(input, img)
    text = 'testing'
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.imread(img_path)
    drawTime(frame, img_path)
    drawTitle(frame, 'Planth Growth Timelapse')
    #cv2.putText(frame, text, (50, 50), font, 2, (255, 255, 0), 2)
    out.write(frame)
    bar.next()

  out.release()
  bar.finish()

if __name__ == "__main__":
  make_video('/home/pi/git/Camera_App/sample_images/', '../output/project.avi', 1)
  #make_heatmap()
