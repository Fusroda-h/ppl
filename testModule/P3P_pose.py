import numpy as np
import poselib
import math
import cv2

pi = math.pi

def error_r(r_pred, r_gt):
    return np.arccos((np.trace(np.dot(np.transpose(r_pred), r_gt)) - 1) / 2)

def error_t(r_pred, r_gt, t_pred, t_gt):
    return np.linalg.norm(np.subtract(np.dot(np.transpose(r_pred), t_pred), np.dot(np.transpose(r_gt), t_gt)))

def get_quaternion_from_euler(roll, pitch, yaw):
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qw, qx, qy, qz]

def convert_to_matrix(q):
    q0, q1, q2, q3 = q
    R = np.zeros((3, 3),dtype=float)
    R[0][0] = 1 - 2*(q2*q2 + q3*q3)
    R[0][1] = 2*(q1*q2 - q3*q0)
    R[0][2] = 2*(q1*q3 + q2*q0)
    
    R[1][0] = 2*(q1*q2 + q3*q0)
    R[1][1] = 1 - 2*(q1*q1 + q3*q3)
    R[1][2] = 2*(q2*q3 - q1*q0)
    
    R[2][0] = 2*(q1*q3 - q2*q0)
    R[2][1] = 2*(q2*q3 + q1*q0)
    R[2][2] = 1 - 2*(q1*q1 + q2*q2)
    return R

def projection(pts,q,center):
  R = convert_to_matrix(q)
  t = - R.T@center.T
  n = pts.shape[0]
  pts = (R@pts.T).T + np.tile(t,[n,1])

  return np.array([pts[:,0]/pts[:,2],pts[:,1]/pts[:,2]]).T, R, t


if __name__ == "__main__":
  pts = np.random.rand(100,3)
  cam_center = np.array([0.5,0.5,-3])
  q = get_quaternion_from_euler(math.pi/4,math.pi/3,math.pi/4)
  
  proj, R, t= projection(pts, q, cam_center)
  
  proj[20:30] = np.random.randn(10,2)

  cam = {'model':'SIMPLE_PINHOLE',
                  'width':1,'height':1,
                  'params':[1,0,0]}
  
  res = poselib.estimate_absolute_pose(proj, pts, cam,{}, {})

  print(proj, pts)

  # predicted RT
  _q = res[0].q
  _t = res[0].t
  _R = convert_to_matrix(_q)
  print(res)
  print('-'*30)
  print('Rotation matrix :\n', _R,'\ntranslation :',_t)
  print('Camera center :',-_R.T@_t)
  print('Quternion :',_q)
  print('-'*30)
  print('-'*30)
  print('GR Rotation matrix:\n',R)
  print('GT Translation :' ,t)
  print('GT Camera center :' ,cam_center)

  print("Rot eror: ", error_r(_R, R))
  print("Trans error: ", error_t(_R, R, _t, t))


