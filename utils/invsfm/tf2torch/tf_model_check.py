import tensorflow as tf
import numpy as np

# Base Model
class Net(object):

    def __init__(self):
        self.weights = {}
    
    # Save weights to an npz file
    def save(self,sess,fname):
        wts = {k:sess.run(v) for k,v in self.weights.items()}
        np.savez(fname,**wts)
        return wts
    
    # Load weights from an npz file
    def load(self,sess,fname=None):
        wts = np.load(fname)
        ops = [v.assign(wts[k].astype(np.float32)).op
               for k,v in self.weights.items() if k in wts]
        if len(ops) > 0:
            sess.run(ops)

    # Get all trainable weights
    def trainable_variables(self):  
        return {v:k for k,v in self.weights.items() if v in tf.trainable_variables()}

            
# Base Model for VisibNet, CaarseNet and RefineNet
class InvNet(Net):
    def __init__(self, inp,
                 bn='train',
                 ech = [256,256,256,512,512,512],
                 dch = [512,512,512,256,256,256,128,64,32,3],
                 skip_conn = 6,
                 conv_act = 'relu',
                 outp_act = 'tanh'):

        super().__init__()
        self.bn = bn
        self.weights = {}
        self.ifdo = tf.Variable(False,dtype=tf.bool)
        self.set_ifdo = self.ifdo.assign(True).op
        self.unset_ifdo = self.ifdo.assign(False).op
        
        # 임의로 추가한 부분
        self.asset = [inp]
        
        #Encoder
        out = inp; skip = [out]
        for i in range(len(ech)):
            out = self.conv(out,4,ech[i],2,True,1.,conv_act,'ec%d'%i)
            skip.append(out)
            # 임의로 추가한 부분
            self.asset.append(out)
        skip = list(reversed(skip))[1:]
                
        # Decoder
        for i in range(len(dch)-1):
            if i<len(ech): out = tf.image.resize_images(out,tf.shape(out)[1:3]*2,method=1)
            out = self.conv(out,3,dch[i],1,True,.5 if i<3 else 1.,conv_act,'dc%d'%i)
            self.asset.append(out)
            if i<skip_conn: out = tf.concat((skip[i],out),axis=3)
        self.pred = self.conv(out,3,dch[-1],1,False,1.,outp_act,'dc%d'%(len(dch)-1))
        # 임의로 추가한 부분
        self.asset.append(self.pred)
        
    # Covolutional layer with Batchnorm, Bias, Dropout  & Activation
    def conv(self,inp,ksz,nch,stride,bn,rate,act,nm):

        # Conv
        ksz = [ksz,ksz,inp.get_shape().as_list()[-1],nch]
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        self.weights['%s_w'%nm] = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        out = tf.pad(inp,[[0,0],[1,1],[1,1],[0,0]],'REFLECT')
        out = tf.nn.conv2d(out,self.weights['%s_w'%nm],[1,stride,stride,1],'VALID')

        # Batchnorm
        if bn:
            if self.bn=='train' or self.bn=='set':
                axis = list(range(len(out.get_shape().as_list())-1))
                wmn = tf.reduce_mean(out,axis)
                wvr = tf.reduce_mean(tf.squared_difference(out,wmn),axis)
                out = tf.nn.batch_normalization(out,wmn,wvr,None,None,1e-3)

                if self.bn=='set':
                    self.weights['%s_mn'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
                    self.weights['%s_vr'%nm] = tf.Variable(tf.ones([nch],dtype=tf.float32))
                    self.bn_outs['%s_mn'%nm] = wmn
                    self.bn_outs['%s_vr'%nm] = wvr
                    
            if self.bn=='test':
                self.weights['%s_mn'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
                self.weights['%s_vr'%nm] = tf.Variable(tf.ones([nch],dtype=tf.float32))
                out = tf.nn.batch_normalization(out,self.weights['%s_mn'%nm],
                                                self.weights['%s_vr'%nm],None,None,1e-3)
                
        # Bias
        self.weights['%s_b'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
        out = out + self.weights['%s_b'%nm]

        # Dropout
        if rate < 1:
            out = tf.cond(self.ifdo, lambda: tf.nn.dropout(out,rate), lambda: out)
        
        # Activation
        if act=='relu':
            out = tf.nn.relu(out)
        elif act=='lrelu':
            out = tf.nn.leaky_relu(out)
        elif act=='sigm':
            out = tf.nn.sigmoid(out)
        elif act=='tanh':
            out = tf.nn.tanh(out)
            
        return out

    
# VisibNet 
class VisibNet(InvNet):
    
    def __init__(self,inp,bn='train',outp_act=True):

        if inp.get_shape().as_list()[-1] < 5:
            ech = [64,128,256,512,512,512]
        else:
            ech = [256,256,256,512,512,512]

        super().__init__(inp,bn=bn,
                         ech = ech,
                         dch = [512,512,512,256,256,256,128,64,32,1],
                         skip_conn = 6,
                         conv_act = 'relu',
                         outp_act = 'sigm' if outp_act else None)
