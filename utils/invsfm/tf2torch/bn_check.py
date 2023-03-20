import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

# pytorch와 tensorflow 동시 구현
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# tensorflow v2 비활성화, v1 활성화
tf.disable_v2_behavior()

# def batchnorm2D(input, mean, var, bias, eps=1e-3):
#     first_term = input/np.sqrt(var+eps)
#     second_term = mean/np.sqrt(var+eps)
#     return first_term-second_term + bias

# class Batchnorm(nn.Module):
#     '''
#     input : NCHW
#     bias : C
#     '''
#     def __init__(self, num_features, mean, var, bias, eps=1e-3, device=None, dtype=None) -> None:
#         factory_kwargs = {'device':device, 'dtype':dtype}
#         super(Batchnorm, self).__init__()
#         self.num_features = num_features
#         self.mean_ = mean
#         self.var_ = var
#         self.bias = bias
#         self.eps = eps

#     def forward(self, input:torch.Tensor) -> torch.Tensor:
#         first_term = input/torch.sqrt(self.var_ + self.eps)
#         second_term = self.mean_/torch.sqrt(self.var_ + self.eps)
#         return first_term - second_term + self.bias

# # tensorflow input data format : NHWC
# tf_inp = tf.placeholder(tf.float32,shape=[1,5,5,3])
# np_inp = np.ones((5,5,5,3), dtype=np.float32)*4
# # pytorch input data format : NCHW
# np_inp2 = np.ones((5,3,5,5))*4
# torch_inp = torch.from_numpy(np_inp2).to(torch.float32)

# # tensorflow batchnorm
# mn = tf.placeholder(tf.float32, shape=[3])
# var = tf.placeholder(tf.float32, shape=[3])
# # np_mn = np.random.randn(3).astype(np.float32)
# # np_var = np.random.rand(3).astype(np.float32)
# np_mn = np.ones(3, dtype=np.float32)*2
# np_var = np.ones(3, dtype=np.float32)*4
# bias = np.array([1.,2.,3.])*4
# eps = 5

# tf_bias = tf.placeholder(tf.float32, shape=[3])
# tf_out_ = tf.nn.batch_normalization(tf_inp, mn, var, None, None, eps) + tf_bias
# sess = tf.Session()
# tf_outs = []
# for i in range(5):
#     fd = {tf_inp:np_inp[i:i+1],
#         mn:np_mn,
#         var:np_var,
#         tf_bias:bias}
#     tf_out = sess.run(tf_out_, feed_dict=fd)
#     tf_outs.append(tf_out)
# tf_out = np.transpose(np.squeeze(np.array(tf_outs), axis=1), (0,3,1,2))
# print('Tensorflow output')
# print(tf_out[0,:,0,0])

# # pytorch batchnorm
# mn_ = torch.from_numpy(np_mn)
# var_ = torch.from_numpy(np_var)
# torch_bias = torch.from_numpy(bias).to(torch.float32)
# bias = np.expand_dims(bias, axis=0)
# bias = np.expand_dims(bias, axis=2)
# bias = np.expand_dims(bias, axis=3)
# bias = np.concatenate([bias, bias, bias, bias, bias], axis=0)
# bias = np.concatenate([bias, bias, bias, bias, bias], axis=2)
# bias = np.concatenate([bias, bias, bias, bias, bias], axis=3)
# torch_out = F.batch_norm(torch_inp, running_mean=mn_, running_var=var_, weight=None, bias=torch_bias, training=False, eps=eps, momentum=0.)
# torch_out = torch_out.numpy()
# '''
# tf.nn.batch_normalization은 단순히 batch_norm만 할 수 있음.

# * affine *
# affine = False로 하면 gamma와 beta가 없는 셈
# 즉 true batch_norm을 할 수 있다.

# * running_track *
# running_mean과 running_var를 쓸 것인지 확인
# False면 None으로 주어지게 됨

# * momentum *
# mean(x_new) = (1-momentum)*x_old_mean + momentum*x_new_mean
# momentum이 0이면 old_mean, 즉 running_mean만 사용하게 됨
# '''


# '''
# 가정1.
# tf.nn.batch_normalization에서 output = scale*(x-mean)/sd + offset 으로 표현됨
# 그렇다면 tf.nn.batch_normalization에서는 output = scale*(x-mean)/(sqrt(variance)+eps) + offset으로 적용되는 것이 아닐까?
# pytorch에서는 output = scale*(x-mean)/(sqrt(variance + eps)) + offset으로 적용되는 것으로 보임

# x=4, mean=2, var=4, eps=5로 설정한 후 실험해보자
# 이 가정이 맞다면 tf_output = 2/7이 되어야 하며, torch_out=2/3이 되어야 한다.
# 2/7 : 0.28571428571428571428571428571429
# 2/3 : 0.66666666666666666666666666666667

# 실험 결과 tf_out 또한 0.6666666이며, torch_out도 0.6666667이 나온다.
# '''
# nnBatchNorm = nn.BatchNorm2d(3, eps=eps, momentum=0., affine=True, track_running_stats=False)
# nnBatchNorm.eval()
# torch_out2 = nnBatchNorm(torch_inp).detach().numpy() + bias
# print('Test something')
# print(torch_out2[0,:,0,0])
# # nnBatchNorm.running_mean = mn_
# # nnBatchNorm.running_var = var_
# # torch_out2 = nnBatchNorm(torch_inp).detach().numpy() + bias
# print(nnBatchNorm.running_mean)
# print(nnBatchNorm.running_var)
# print(nnBatchNorm.weight)
# print(nnBatchNorm.bias)
# print('Pytorch output')
# print(torch_out[0,:,0,0])
# print('Test something')
# print(torch_out2[0,:,0,0])
# torch_bias = nn.Parameter(torch_bias)
# nnBatchNorm.bias = torch_bias
# torch_out3 = nnBatchNorm(torch_inp)
# torch_out3 = torch_out3.detach().numpy()
# print('Test something else')
# print(torch_out3[0,:,0,0])
# diff = torch_out - tf_out
# # print(diff[0][0])


########
tf_inp = tf.placeholder(tf.float32,shape=[1,3,3,3])
print(len(tf_inp.get_shape().as_list()))
axis = list(range(len(tf_inp.get_shape().as_list())-1))
wmn = tf.reduce_mean(tf_inp,axis)
wvr = tf.reduce_mean(tf.squared_difference(tf_inp,wmn),axis)
out = tf.nn.batch_normalization(tf_inp,wmn,wvr,None,None,1e-3)
sess = tf.Session()
np_inp = np.reshape(np.random.rand(27), (1,3,3,3))
tf_outs=[]
tf_inps=[]
axiss = []
wmns = []
wvrs = []
print(axis)
for i in range(1):
    fd = {tf_inp:np_inp[i:i+1]}
    tf_out = sess.run(out, feed_dict=fd)
    tf_inp = sess.run(tf_inp, feed_dict=fd)
    wmn_ = sess.run(wmn, feed_dict=fd)
    wvr_ = sess.run(wvr, feed_dict=fd)
    tf_outs.append(tf_out)
    tf_inps.append(tf_inp)
    wmns.append(wmn_)
    wvrs.append(wvr_)
tf_inp = np.transpose(np.squeeze(np.array(tf_inps), axis=1), (0,3,1,2))
tf_out = np.transpose(np.squeeze(np.array(tf_outs), axis=1), (0,3,1,2))
print('Tensorflow input')
print(tf_inp)

print('\nwmn')
print(wmns)
print('\nwvr')
print(wvrs)
print('\nTensorflow output')
print(tf_out)

