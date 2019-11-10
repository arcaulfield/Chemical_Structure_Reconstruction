
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA


# In[5]:


df = pd.read_csv('descriptors.csv')
df.shape


# In[6]:


# Separating out & normalizing the features
x = df.iloc[:, 1:1404].values
x = preprocessing.scale(x)

# Separating out the target
y = df.iloc[:,0].values


# In[7]:


# features = list(df.columns)
# features.pop(0)
targets = list(df['ID'])
#targets


# In[8]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])


# In[ ]:


##### TEST: How many components to select ##### 


# In[185]:


### Testing 3 components ### 
    # pca = PCA(n_components=3)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])
    # pca.explained_variance_ratio_
        # OUPUT: array([ 0.23354774,  0.11475039,  0.06810382])


# In[186]:


### Testing 5 components ###
    # pca = PCA(n_components=5)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3','PC4','PC5'])
    # pca.explained_variance_ratio_
        # OUTPUT: array([ 0.23354774,  0.11475039,  0.06810382,  0.0412066 ,  0.03132345])


# In[187]:


### Testing 7 components ### 
    # pca = PCA(n_components=7)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','PC7'])
    # pca.explained_variance_ratio_
        # OUTPUT: array([ 0.23354774,  0.11475039,  0.06810382,  0.04120661,  0.03132366,
            #0.02704538,  0.02453726])


# In[ ]:


### Conclusion ### 
    # We selected the first two principal components (PCs) because the additional components
    # explain less than 7% of the variance 
    # (PC1: 23.4% of the variance; P2: 11.5% of the variance).


# In[9]:


finalDf = pd.concat([principalDf, df[['ID']]], axis = 1)


# In[13]:


### PLOT: 2 Component PCA ###
        # x-axis = PC1, y-axis=PC2, data point = how much COMPOUND (NOT feature) is influenced by PCs 1 and 2
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('2-Component PCA', fontsize = 20)
for target in targets:
    indicesToKeep = finalDf['ID'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               #, c = color
               , s = 50)
ax.grid()


# In[10]:


### Variance explained by each component ###
pca.explained_variance_ratio_
    # OUTPUT: array([ 0.23354774,  0.11475039])


# In[11]:


### FUNCTION to generate heat map of each PC explained by descriptors ###

def heat_map(ar, n,m):
#     a = ar[1]
#     b = ar[2]
    #c = ar[1:2,:]
    #c = np.asarray(c)
    c = np.delete(ar,0,0)
    c = c.astype(np.float)
    plt.matshow(c,cmap='viridis',vmin=n,vmax=m)
    plt.yticks([0,1],['PC1','PC2'],fontsize=10)
    t = c.shape[1]
    
    plt.xticks(range(int(t)), ar[0])
    plt.colorbar()
    plt.show()

### Example: first 10 descriptors ###
b = np.asarray(pca.components_[0,0:10])
c = np.asarray(pca.components_[1,0:10])
a = df.columns[1:11]
d = [a,b,c]
heat_map(d, 0, 1)


# In[12]:


### How much each feature contributes to each component
pca.components_
a = np.asarray(pca.components_)
a.shape
#pd.DataFrame(pca.components_).to_csv("bla.csv")


# In[13]:


##### Important Feature Extraction #####
    # Function: Thresholds & Displays the Data
    # c: array where rows = components, columns = descriptors, entries = contribution of descriptor to respective component
def data_threshold(c,n, min, max):
    threshold = n
    d = c[1,:].astype(float) + c[2,:].astype(float)
    d=d.reshape(1,1403)
    d = np.vstack((d, d, d))
    e=c[d > n]
    e=np.asarray(e)
    t = e.shape[0]

    # number of descriptors that respect the threshold
    t= t/3
    print(t)

    f=(e[0:int(t)])
    g=(e[int(t):int(2*t)])
    h=(e[int(2*t):int(3*t)])

        # creating array with only columns from c that respect threshold!
    final = np.vstack((f, g, h))
    ###print(final.shape)
    pd.DataFrame(final).to_csv("descript_"+str(n)+".csv")
    heat_map(final,min,max)


# In[14]:


#### TEST: Thresholding & Displaying Data
features = list(df.columns)
features.pop(0)
features
b = []
b.append(features)
b.append(a[0])
b.append(a[1])
c = np.asarray(b)
c.shape
data_threshold(c, 0.08,0,0.08)

