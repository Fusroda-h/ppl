import numpy as np
import poselib
import math
import cv2
import os
import sys

cur_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(os.path.join(cur_dir, "../"))

from utils.pose.pose_estimation import *

def main():
  pts = np.random.rand(100,3)
  cam_center = np.array([0.5,0.5,-3])
  q = get_quaternion_from_euler(math.pi/4,math.pi/3,math.pi/4)
  
  proj, R, t= projection(pts, q, cam_center)
  
  cam = {'model':'SIMPLE_PINHOLE',
                  'width':1,'height':1,
                  'params':[1,0,0]}
  
  res = poselib.estimate_absolute_pose(proj,pts,cam,{}, {},1)

  _q = res[0].q
  _t = res[0].t
  _R = convert_to_matrix(_q)
  print("Absolute Pose Error")
  print("Rot eror: ", error_r(_R, R))
  print("Trans error: ", error_t(_R, R, _t, t))
  print("***********************************")

  lines = np.random.rand(100,3)
  res = poselib.estimate_p6l_relative_pose(proj, pts, lines, [cam], [cam], {}, {}, 1)

  _q = res[0].q
  _t = res[0].t
  _R = convert_to_matrix(_q)
  print("relative Pose Error")
  print("Rot eror: ", error_r(_R, R))
  print("Trans error: ", error_t(_R, R, _t, t))
  print("***********************************")



if __name__ == "__main__":
  main()

 