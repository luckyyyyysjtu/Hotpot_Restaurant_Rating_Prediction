#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
filename = 'C:/Users/30265/Desktop/iris.csv'
dataframe = pd.read_csv(filename)
dataset = np.array(dataframe)


# In[83]:


dataset


# In[86]:


input=dataset[:,1:-3]
y=dataset[:,-3:]


# In[87]:


input


# In[88]:


y


# In[89]:


from sklearn import preprocessing
max_abs_scaler = preprocessing.MaxAbsScaler()
input = max_abs_scaler.fit_transform(input)


# In[90]:


input


# In[91]:


def nueronnetwork(input,y,q,l,rate):
    global v,w
    b=np.ones(input.shape[0])
    input=np.c_[input,b]
    d=input.shape[1]
    m=input.shape[0]

    v=np.random.random((d, q))  #第一层系数矩阵初始化
    w=np.random.random((q,l)) #第二层系数矩阵初始化
    print(w)
 

    for round in range(100):
        sumoferr=0
        for i in range(m):
            row = input[i, :]
            hidden_layer_input = np.dot(row, v)
            hidden_layer_output = 1 / (1 + np.e ** ((-1) * hidden_layer_input))
            output = np.dot(hidden_layer_output, w)
            yhat = 1 / (1 + np.e ** ((-1) * output))
            g = yhat * (1 - yhat) * (y[i,:] - yhat)
            e = hidden_layer_output * (1 - hidden_layer_output) * np.dot(g, np.transpose(w))
            # 计算delw和delv
            delw = np.zeros((q, l))
            for c in range(q):
                for f in range(l):
                    delw[c, f] = hidden_layer_output[c] * g[f] * rate
            delv = np.zeros((d, q))
            for x in range(d):
                for z in range(q):
                    delv[x, z] = input[i,x] * e[z] * rate
            # 更新权值
            w = w + delw
            v = v + delv
            error=np.dot(y[i,:]-yhat,y[i,:]-yhat)*0.5
            #print('Error for observation%d is %f' %(i,error))
            sumoferr+=error 

        print('Round %d: Error for the whole dataset is %f'%(round,sumoferr))

    


# In[92]:


def predict(data):
    b=np.ones(input.shape[0])
    data=np.c_[data,b]
    predicted=list()
    for i in range(data.shape[0]):
        row = data[i, :]
        hidden_layer_input = np.dot(row, v)
        hidden_layer_output = 1 / (1 + np.e ** ((-1) * hidden_layer_input))
        output = np.dot(hidden_layer_output, w)
        yhat = 1 / (1 + np.e ** ((-1) * output))
        y=max(yhat)
        num=yhat.tolist().index(y)
        predicted.append(num)
    
        
    return predicted


# In[93]:


nueronnetwork(input,y,4,3,0.05)


# In[94]:


predict(input)


# In[103]:


import matplotlib.pyplot as plt 
plt.hist(y-predict(input))
plt.show()


# In[101]:


y=dataset[:,-1]
y

