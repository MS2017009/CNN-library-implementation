#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[2]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
def get_batch(X, y, batch_size=1000):
    for i in np.arange(0, y.shape[0], batch_size):
        end = min(X.shape[0], i + batch_size)
        yield(X[i:end],y[i:end])


# ConVolution Function

# In[3]:


def convolution(batch,kernels):
    stride=(1,1,1)
    stride=np.asarray(stride)
    blocks = (np.array(batch[0,:,:].shape)-np.array(kernels[0,:,:].shape))//np.array(stride)+1
    #print blocks
    no_blocks_row_image=blocks[0]
    no_blocks_col_image=blocks[1]
    total_blocks=(no_blocks_row_image)*(no_blocks_col_image)
    #print total_blocks
    dim1=len(kernels[0,:,0,0])*len(kernels[0,0,:,0])*total_blocks
    dim2=len(kernels)
    dim3=len(batch)
    #print dim3,dim1,dim2
    def f(i,j,k):
        block_no=j//(len(kernels[0,:,0,0])*len(kernels[0,0,:,0]))
        row_kernel=(j%(len(kernels[0,:,0,0])*len(kernels[0,0,:,0])))//len(kernels[0,:,0,0])
        col_kernel=(j%(len(kernels[0,:,0,0])*len(kernels[0,0,:,0])))%len(kernels[0,:,0,0])
        row_image=((block_no//no_blocks_row_image)*stride[0])+row_kernel
        col_image=((block_no%no_blocks_col_image)*stride[1])+col_kernel
        cache1=[]
        cache1.append(row_image)
        cache1.append(col_image)
        cache1=np.asarray(cache1)
        convolv = np.sum(batch[i,row_image,col_image,:]*kernels[k,row_kernel,col_kernel,:],axis=3)
        convolv=np.reshape(convolv,(dim3,total_blocks,(len(kernels[0,:,0,0])*len(kernels[0,0,:,0])),dim2))
        convolv=np.sum(convolv,axis=2)
        convolv=np.reshape(convolv,(dim3,blocks[0],blocks[1],dim2))
        return convolv,cache1   
    convolved_images=np.fromfunction(f,(dim3,dim1,dim2),dtype=int)
    return convolved_images
    


# Relu Activation Function

# In[4]:


def relu(image_layer1):
    shape=image_layer1.shape
    def f(i,j,k,l):
        if(image_layer1[i,j,k,l].all<0.0):
            image_layer1[i,j,k,l]=0.0
        return image_layer1[i,j,k,l]
    relued_images=np.fromfunction(f,(shape[0],shape[1],shape[2],shape[3]),dtype=int)
    return relued_images


# Average Pooling Function

# In[5]:


def pooling(image_layer_relu,size,stride):
    blocks_pooled_row=int(((len(image_layer_relu[0,:,0,0])-size)//stride)+1)
    blocks_pooled_col=((len(image_layer_relu[0,0,:,0])-size)//stride)+1
    total_blocks_pooled=blocks_pooled_row*blocks_pooled_col
    dim3=len(image_layer_relu)
    dim1=size*size*total_blocks_pooled
    dim2=len(image_layer_relu[0,0,0])
    def f(k,i,j):
        block_no=i//(size*size)
        row_block=(i%(size*size))//size
        col_block=((i%(size*size))%size)
        row_image=((block_no//blocks_pooled_row)*stride)+row_block
        col_image=((block_no%blocks_pooled_col)*stride)+col_block
        pooled=image_layer_relu[k,row_image,col_image,j]
        pooled=np.reshape(pooled,(dim3,total_blocks_pooled,size*size,dim2))
        indices=np.argmax(pooled,axis=2)
        pooled=np.average(pooled,axis=2)
        pooled=np.reshape(pooled,(dim3,blocks_pooled_row,blocks_pooled_col,dim2))
        x_reshaped = image_layer_relu.reshape(dim3, dim2,blocks_pooled_row, 2, blocks_pooled_col,2)
        cache=(image_layer_relu,x_reshaped,pooled)
        return pooled,cache
    
    image_layer_pooled,cache=np.fromfunction(f,(dim3,dim1,dim2),dtype=int)
    
    return image_layer_pooled,cache


# Pooling  Backward Function

# In[6]:


def pooled_backward(image_layer_relu_reshape,size,stride,total_blocks_pooled,dx_reshaped):
    dim3=image_layer_relu_reshape.shape[0]
    dim1=image_layer_relu_reshape.shape[1]
    dim2=image_layer_relu_reshape.shape[2]
    
    def f(k,i,j):    
        n=i//(size*size)
        dx_relu=image_layer_relu_reshape[k,i,j]+dx_reshaped[k,n,j]
        return dx_relu
    previous_images_reshaped=np.fromfunction(f,(dim3,dim1,dim2),dtype=int)
    return previous_images_reshaped


# Relu Backward Function
# 

# In[7]:


def relu_backward(dx_relu):
    dx_relu[dx_relu<=0]=0
    dx_relu[dx_relu>0]=1
    dx_conv=dx_relu
    return dx_conv


# Convolution Backward

# In[8]:


def conv_backward(dx_conv_repeat,batch,kernels):
    stride=(1,1,1)
    stride=np.asarray(stride)
    blocks = (np.array(batch[0,:,:].shape)-np.array(kernels[0,:,:].shape))//np.array(stride)+1
    #print blocks
    no_blocks_row_image=blocks[0]
    no_blocks_col_image=blocks[1]
    total_blocks=(no_blocks_row_image)*(no_blocks_col_image)
    dim1=len(kernels[0,:,0,0])*len(kernels[0,0,:,0])*total_blocks
    #print dim1
    dim2=len(kernels)
    dim3=len(batch)
    
   
    shape_kernel=kernels.shape
    def f(i,j):
        block_no=j//(len(kernels[0,:,0,0])*len(kernels[0,0,:,0]))
        row_kernel=(j%(len(kernels[0,:,0,0])*len(kernels[0,0,:,0])))//len(kernels[0,:,0,0])
        col_kernel=(j%(len(kernels[0,:,0,0])*len(kernels[0,0,:,0])))%len(kernels[0,:,0,0])
        row_image=((block_no//no_blocks_row_image)*stride[0])+row_kernel
        col_image=((block_no%no_blocks_col_image)*stride[1])+col_kernel
        
        d=batch[i,row_image,col_image,:]
        #print d.shape
        return d
    df_input=np.fromfunction(f,(dim3,dim1),dtype=int)
    def g(i,j,k):
        n=j//9
        da=np.sum(df_input[i,j,:]*dx_conv_repeat[i,n,:,k],axis=3)
        return da
    da_input=np.fromfunction(g,(dim3,dim1,dim2),dtype=int)
    da_input=np.sum(np.reshape(da_input,(dim3,len(kernels[0,:,0,0])*len(kernels[0,0,:,0]),total_blocks,dim2)),axis=2)
    da_input=np.reshape(da_input,(dim3,len(kernels[0,:,0,0]),len(kernels[0,0,:,0]),dim2))
    return da_input


# Calling batch of 1000 images at time and doing forward and Backward Propogation

# In[9]:


#print X_train.shape
#print X_test.shape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#W=np.random.random((pooled_reshaped.shape[1],10))*0.00001
kernel1=np.random.rand(3,3,1)*0.0001
kernel2=np.random.rand(3,3,1)*0.0001
kernel3=np.random.rand(3,3,1)*0.0001
kernels=[]
kernels.append(kernel1)
kernels.append(kernel2)
kernels.append(kernel3)
kernels=np.asarray(kernels)
#print kernels.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
i=0

for (batch , y) in get_batch(X_train, y_train, batch_size=1000):
    #batch_index=np.random.choice(60000,100)
    #batch=X_train[batch_index]
    cv2.imwrite("actual_image.jpg",batch[0])
    #y=y_train[batch_index]
    images_layer1,cache1=convolution(batch,kernels) 
    #print images_layer1.shape
    cv2.imwrite("image_layer1.jpg",images_layer1[0,:,:,0])
    image_layer_relu=relu(images_layer1)
    #print image_layer_relu.shape
    cv2.imwrite("relu.jpg",image_layer_relu[0,:,:,0])

    shape=np.array(image_layer_relu.shape)
    size=2
    stride=2
    pooled_images,cache_pooled=pooling(image_layer_relu,size,stride)
    cv2.imwrite('pooled.jpg',pooled_images[99,:,:,0])
    #print pooled_images.shape
    pooled_reshaped=np.reshape(pooled_images,(pooled_images.shape[0],(pooled_images.shape[1]*pooled_images.shape[2]*pooled_images.shape[3])))
    #print pooled_reshaped.shape
    X=pooled_reshaped
    if(i==0):
        W=np.random.random((pooled_reshaped.shape[1],10))*0.00001
    
    i=1
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    f = X.dot(W)
    shifted_logits = f - np.max(f, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = f.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    print loss
    df = probs.copy()
    df[np.arange(N), y] -= 1
    df /= N
    df.shape

    dx = df.dot(W.T).reshape(X.shape)
    dw = X.reshape(X.shape[0],-1).T.dot(df)
    dx_reshaped=np.reshape(dx,(pooled_images.shape[0],pooled_images.shape[1],pooled_images.shape[2],pooled_images.shape[3]))
    dx_reshaped=dx_reshaped/4
    dx_reshaped=np.reshape(dx_reshaped,(pooled_images.shape[0],pooled_images.shape[1]*pooled_images.shape[2],pooled_images.shape[3]))
    alpha=0.5
    

    blocks_pooled_row=int(((len(image_layer_relu[0,:,0,0])-size)//stride)+1)
    blocks_pooled_col=((len(image_layer_relu[0,0,:,0])-size)//stride)+1
    total_blocks_pooled=blocks_pooled_row*blocks_pooled_col
    dim3=len(image_layer_relu)
    dim1=size*size*total_blocks_pooled
    dim2=len(image_layer_relu[0,0,0])
    image_layer_relu_reshape=np.reshape(image_layer_relu,(dim3,dim1,dim2))
    dx_relu=pooled_backward(image_layer_relu_reshape,size,stride,total_blocks_pooled,dx_reshaped)
    dx_conv=relu_backward(dx_relu)
    #print dx_conv.shape
    
    dx_conv_repeat=np.repeat(dx_conv[:, :, np.newaxis,:], batch.shape[3], axis=2)
    #print dx_conv_repeat.shape
    da_input=conv_backward(dx_conv_repeat,batch,kernels)
    #print da_input.shape
    dkernel= da_input[9]
    #print dkernel[0]
    #dkernel=np.repeat(dkernel[:,:,:,np.newaxis],3,axis=0)
    dkernel=np.reshape(dkernel,(dkernel.shape[0],dkernel.shape[1],dkernel.shape[2],1))
    c=W-(alpha*dw)
    W=c
    alpha=0.5
    #print dkernel[0]
    w1=kernels[0]-(alpha*dkernel[0])
    w2=kernels[1]-(alpha*dkernel[1])
    w3=kernels[2]-(alpha*dkernel[2])
    kernel=[]
    kernel.append(w1)
    kernel.append(w2)
    kernel.append(w3)
    kernel=np.asarray(kernels)
    kernels=kernel
      
    


# In[ ]:





# In[ ]:





# In[ ]:




