
# coding: utf-8

# # Assignment 2 (Practical)
# 
# **COMP9418 - Advanced Topics in Statistical Machine Learning**
# 
# **Louis Tiao** (TA), **Edwin V. Bonilla** (Instructor)
# 
# *School of Computer Science and Engineering, UNSW Sydney*
# 
# ---
# 
# In the practical component of this assignment you will build a *class-conditional classifier* using the mixture model described in the theory section of this assignment.
# 
# The basic idea behind a class conditional classifier is to train a separate model for each class $p(\mathbf{x} \mid y)$, and use Bayes' rule to classify a novel data-point $\mathbf{x}^*$ with:
# 
# $$
# p(y^* \mid \mathbf{x}^*) = \frac{p(\mathbf{x}^* \mid y^*) p(y^*)}{\sum_{y'=1}^C p(\mathbf{x}^* \mid y') p(y')}
# $$
# 
# (c.f. Barber textbook BRML, 2012, $\S$23.3.4 or Murphy textbook MLaPP, 2012, $\S$17.5.4).
# 
# In this assignment, you will use the prescribed mixture model for each of the conditional densities $p(\mathbf{x} | y)$ and a Categorical distribution for $p(y)$.
# 
# ### Prerequisites
# 
# You will require the following packages for this assignment:
# 
# - `numpy`
# - `scipy`
# - `scikit-learn`
# - `matplotlib`
# - `observations`
# 
# Most of these may be installed with `pip`:
# 
#     pip install numpy scipy scikit-learn matplotlib observations
# 
# ### Guidelines
# 
# 1. Unless otherwise indicated, you may not use any ML libraries and frameworks such as scikit-learn, TensorFlow to implement any training-related code. Your solution should be implement purely in NumPy/SciPy.
# 2. Do not delete any of the existing code-blocks in this notebook. It will be used to assess the performance of your algorithm.
# 
# ### Assessment
# 
# Your work will be assessed based on:
# - **[50%]** the application of the concepts for doing model selection, which allows you to learn a single model for prediction (Section 1);  
# - **[30%]** the code you write for making predicitions in your model (Section 2); and
# - **[20%]** the predictive performance of your model (Section 3). 

# ## Dataset
# 
# You will be building a class-conditional classifier to classify digits from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), containing grayscale images of clothing items --- coats, shirts, sneakers, dresses and the like.
# 
# This can be obtained with [observations](https://github.com/edwardlib/observations), a convenient tool for loading standard ML datasets.

# In[ ]:

from observations import fashion_mnist
from sklearn.preprocessing import LabelBinarizer
import sklearn.decomposition
from scipy import stats
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import linalg
# In[ ]:

(x_train, y_train_), _ = fashion_mnist('.')


# There are 60k training examples, each consisting of 784-dimensional feature vectors corresponding to 28 x 28 pixel intensities.

# In[ ]:

# x_train.shape


# The pixel intensities are originally unsigned 8-bit integers (`uint8`) and should be normalized to be floating-point decimals within range $[0,1]$.

# In[ ]:

x_train = x_train / 255.

print (x_train.shape)
reducer=sklearn.decomposition.PCA(n_components=100)
reducer.fit(x_train)
x_train=reducer.transform(x_train)
x_train=x_train[:600]

# print (x_train[0],x_train[1])
print (x_train.shape)
# The targets contain the class label corresponding to each example. For this assignment, you should represent this using the "one-hot" encoding. 

# In[ ]:

y_train = LabelBinarizer().fit_transform(y_train_)
y_train=y_train[:600]
# y_train.shape


# Note that you are only to use the training data contained in `x_train`, `y_train` as we have define it. In order to learn and test you model, you may consider splitting these data into training, validation and testing. You may not use any other data to for training.
# 
# In particular, if you want to assess the performance of your model in section 2, you must create a test set `test.npz`. You are not required to submit this test file as we will evaluate the performance of your model using our own test data.

# ## Preamble 

# In[ ]:




# In[ ]:

# #### Constants

# You can use the function below to plot a digits in the dataset.

# In[ ]:

def plot_image_grid(ax, images, n=20, m=None, img_rows=28, img_cols=28):
    """
    Plot the first `n * m` vectors in the array as 
    a `n`-by-`m` grid of `img_rows`-by-`img_cols` images.
    """
    if m is None:
        m = n
 
    grid = images[:n*m].reshape(n, m, img_rows, img_cols)

    return ax.imshow(np.vstack(np.dstack(grid)), cmap='gray')


# Here we have the first 400 images in the training set.

# In[ ]:

# fig, ax = plt.subplots(figsize=(8, 8))
#
# plot_image_grid(ax, x_train, n=20)
#
# plt.show()
#
#
# # Here we have the first 400 images labeled "t-shirts" in the training set.
#
# # In[ ]:
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# plot_image_grid(ax, x_train[y_train_ == 0])
#
# plt.show()


# ## Section 1 `[50%]`: Model Training
# 
# Place all the code for training your model using the function `model_train` below. 
# 
# - We should be able to run your notebook (by clicking 'Cell->Run All') without errors. However, you must save the trained model in the file `model.npz`. This file will be loaded to make predictions in section 2 and assess the performance of your model in section 3. Note that, in addition to this notebook file, <span style="color:red"> ** you must provide the file `model.npz` **</span>.
# 
# - You should comment your code as much as possible so we understand your reasoning about training, model selection and avoiding overfitting. 
# 
# - You can process the data as you wish, e.g. by applying some additional transformations, reducing dimensionality, etc. However, all these should be here too. 
# 
# - Wrap all your training using the function `model_train` below. You can call all other custom functions within it.

# In[ ]:
def Gaussian(data,mean,cov):
    dim = np.shape(data)[0]
    # print('weidu',dim)# 计算维度!!!!!!!!!!!!!!!!!
    covdet = np.linalg.det(cov) # 计算|cov|
    # print(covdet)
    covinv = np.linalg.pinv(cov) # 计算cov的逆
    # print(covinv.shape)
    if covdet==0.0 and covdet==-0.0:              # 以防行列式为0
        covdet = np.linalg.det(cov+np.eye(dim)*0.01)
        # covdet = 1
        covinv = np.linalg.pinv(cov+np.eye(dim)*0.01)
        ##矩阵求逆
    diff = data - mean
    z = -0.5 * np.multiply(np.multiply(diff, covinv),diff.T)
    # print('z',np.power(2*np.pi,784))# 计算exp()里的值
    return (2*np.pi)**(dim/2) / (np.sqrt(np.power(2 * np.pi, dim) * abs(covdet))) * np.exp(z)
# def E_z():
def model_train(x_train, y_train):
    """
    Write your code here.
    """

    global dic
    K = y_train.shape[1]  ###10类
    model=[]##保存模型
    for k in range(K):
        data=x_train[y_train[:,k]==1]
        D,N=data.shape##60000,784
        print (D,N)

        pai=np.ones(K)/K
        Q=30
        bias=np.exp(-700)
        ##hidden variable Q*1
        # Z=np.array(np.random.normal(loc=0,scale=0.1,size=Q).reshape([Q,1]))##对于隐变量
        ##mean N*1
        miu=np.array([np.mean(data,axis=0)]*K).reshape(K,N,1)
        ##Factor Loading W N*Q
        scale = np.power(np.linalg.det(np.cov(data)), (1 / N))
        W = np.array(np.random.randn(K,N,Q))*np.sqrt(scale/Q)
        W_and_miu_new=np.array(np.zeros(shape=[K,N,Q+1]))
        # for k in  range(K):
        #     W_and_miu_new[k] = np.column_stack((W[k], miu[k]))
        ##variance D
        psi=np.diag(np.cov(data,rowvar=False))+bias
        print ('dasas',psi.shape)#####维度为（100，）
        ##Beta  K##
        beta=np.zeros(shape=[K,Q,N])
        smooth = 0.1 * np.eye(100, M=None, k=0)
        # print (beta)
        const=(2*np.pi)**(-D/2)

        # print (scale)
        newloglikelyhood=0
        oldloglikelyhood=1001
        Ez_w_x=np.zeros(shape=[D,K,Q,1])#####60000*10*Q
        Ezz_w_x=np.zeros(shape=[D,K,Q,Q])####Q*10*Q
        Ez_w_x_2 = np.zeros(shape=[D, K, Q+1, 1])
        Ezz_w_x_2 = np.zeros(shape=[D, K, Q+1, Q+1])
        rnk = np.array([np.zeros(K) for i in range(D)])###初始rnk表   60000*10
        # print (rnk.shape)
        # while np.abs(oldloglikelyhood-newloglikelyhood)>0.0001:  ###10类
        # while np.abs(oldloglikelyhood-newloglikelyhood)>500:
        for ite in range(10):
            # oldloglikelyhood=newloglikelyhood
            print ('迭代')

            ##-----------EEEE-step----------------##
            ##get responsibility of all data##
            for  i in range(D):
                for k in range(K):
                    # print (np.matmul(W[k],W[k].T).shape,psi.shape)
                    cov=np.matmul(W[k],W[k].T)+np.diag(psi)

                    # print (data[i].reshape(data[i].shape[0],1),miu[k].shape)
                    mean=data[i].reshape(data[i].shape[0],1)-miu[k]
                    # print(mean.shape)
                    Gaussian=stats.norm.pdf(data[i],mean.reshape(-1),cov)
                    # print(data[i])
                    # print('得出的高斯函数值',Gaussian.pdf(data[i]))
                    rnk[i][k]=pai[k]*np.mean(Gaussian)
                    ##------------------------------------------##
                    ##计算Ez和Ezz
                    tem = psi + np.matmul(W[k], W[k].T)
                    if np.linalg.det(tem) == 0:
                        beta[k] = np.matmul(W[k].T, np.linalg.pinv(tem))
                        # tem[0][0] = tem[0][0] + bias * 0.01
                    else:
                        tem = tem
                        # print (np.matmul(W[k].T, np.linalg.inv(tem)))
                        beta[k] = np.matmul(W[k].T, np.linalg.inv(tem))
                    diff = data[i].reshape(data[i].shape[0],1) - miu[k]
                    # diff = diff.reshape(diff.shape[0], 1)
                    ##calculate E[z|w_k,x_i]
                    Ez_w_x[i][k] = np.matmul(beta[k], (diff))
                    data_i = data[i]
                    # print ('qqqq', data_i.shape)
                    data_i = data_i.reshape(data_i.shape[0], 1)
                    line_one = np.ones(shape=(1, 1))
                    ####Ez-------------------#####
                    Ez_w_x_2[i][k] = np.vstack((Ez_w_x[i][k], line_one))
                    Ezz_w_x[i][k] = (np.identity(Q) - np.matmul(beta[k], W[k]) + np.matmul(np.matmul(np.matmul(beta[k], diff), diff.T),beta[k].T))
                    # print ('E2', Ezz_w_x.shape)
                    ####------------Ezz--------------###
                    Ezz_w_x_2[i][k] = np.column_stack((np.row_stack((Ezz_w_x[i][k], Ez_w_x[i][k].T)), Ez_w_x_2[i][k]))
                    # print('得出',)
            #####------------单独计算W an miu
            W_and_miu_new[k]=np.column_stack((W[k],miu[k]))
            ##计算Q（log_likelihood）--------------------
            # print (rnk)
            sum = 0
            for i in range(D):
                for k in range(K):
                    # print (W_and_miu_new[k].T, np.linalg.pinv(np.diag(psi)))
                    xx = np.matmul(np.matmul(np.matmul(W_and_miu_new[k].T, np.linalg.pinv(np.diag(psi))),W_and_miu_new[k]), Ezz_w_x_2[i][k])
                    p4 = 0.5 * rnk[i][k] * np.trace(xx)
                    p2 = 0.5 * rnk[i][k] * np.matmul(np.matmul(data[i].T, np.linalg.pinv(np.diag(psi))),data[i])
                    # print ('PPPP2',p2)
                    p3 = 1 * rnk[i][k] * np.matmul(
                    np.matmul(np.matmul(data[i].T, np.linalg.pinv(np.diag(psi))), W_and_miu_new[k]),Ez_w_x_2[i][k])
                    p3 = p3
                    sum = p2 - p3 + p4 + sum
            # print (psi)
                # print (np.log(abs(np.linalg.det(np.diag(psi)))))
            p1 = (D / 2) * np.log(abs(np.linalg.det(np.diag(psi))))
            # (2 * np.pi) ** (-D / 2)
            newloglikelyhood = const-p1 - sum
            print('NEWLOG', newloglikelyhood)
            ##现在在一次迭代中我们已经得到###
            ####----Q,Ezz_2,Ez_2,W_and_miu,rnk,psi的矩阵------###
            ##--------M-step----------------########
            for k in range(K):
            ##更新factor loading W and mean miu
                ##跟新pai 对i求和
                W_k_p1_sum = np.zeros(shape=[N,Q+1])
                Mu_k_p1_sum = np.zeros(shape=[Q +1,Q+1])
                pai_new_sum=0

                for i in range(D):
                    W_k_p1_sum=rnk[i][k]*np.matmul(data[i].reshape(data[i].shape[0],1),Ez_w_x_2[i][k].T)+W_k_p1_sum
                    Mu_k_p1_sum=rnk[i][k]*Ezz_w_x_2[i][k]+Mu_k_p1_sum
                    ###pai的加和
                    # print ('RNK',rnk[i][k])
                    pai_new_sum=rnk[i][k]+pai_new_sum
                pai[k]=pai_new_sum/N   #####更新PAI
                # print ('PPPAAAAAIII',pai)
                W_and_miu_new[k]=np.matmul(W_k_p1_sum,np.linalg.pinv(Mu_k_p1_sum))
                # print ('一个NEW',W_and_miu_new.shape)
                W[k,:,:]=W_and_miu_new[k,:,:W_and_miu_new[k].shape[1]-1]
                # print ('XIN WWW',W.shape)####更新WWWWW
                miu[k,:]=W_and_miu_new[k,:,-1].T.reshape(100,1)  ####更新MIU!!
            ##更新协方差矩阵
            psi_new_p0=np.zeros(shape=[N,N])
            ##对i求和
            for i in range(D):
                ##对 k求和，
                data_i=data[i].reshape(data[i].shape[0],1)
                psi_new_p1=np.zeros(shape=[N,N])
                # print (psi_new_p1.shape)
                for k in range(K):
                    pp1=np.matmul(W_and_miu_new[k],Ez_w_x_2[i][k])
                    # print ('P111',p1.shape)
                    psi_new_p1=rnk[i][k]*np.matmul((data_i-pp1),data_i.T)+psi_new_p1
                # print ('qqqqqqqqqq',psi_new_p1.shape)
                psi_new_p0=psi_new_p1+psi_new_p0
                # print (psi_new_p1.shape)
            ##最后的取对角线得新的协方差矩阵
            # print ('%%%%%%%',psi_new_p0.shape)
            #####见论文
            psi=np.diag(psi_new_p0)/D# 更新方差
            print ('PSI',psi.shape)
            # print ('PPPSSSII',Psi_New,np.trace(psi_new_p0))
            # rnk_=rnk/sumres
            #     r.append(np.sum(rnk))##????????????
            # print('每一行数据的和', r)
            # # print('dasdas',len(r))
            # R.append(r)
        # print(np.array(R)[49])

        print('save_model')
        dic={'miu':miu,'pai':pai,'W':W,'psi':psi}
        # print ()
        # const=-N/2*log(np.linalg.det(psi))
        # part2=0
        # # part3=
        # for i in range(N):
        #     for j in range(K):
        #         part2=0.5*rnk*data[i].T*np.linalg.inv(psi)*data[i]+part2

        submodel = dic
        model.append(submodel)
    model=model
    # You can modify this to save other variables, etc 
    # but make sure the name of the file is 'model.npz.
    np.savez_compressed('model.npz', model=model)

# In[ ]:
def model_predict(x_test):

    K=10
    model = np.load('model.npz')
    all_parater=model['model'].tolist()
    # dic={'miu':miu,'psi':psi,'W':W,'pai':pai}
    # print(all_parater[0])
    y_pred=[]
    y_log_prob=[]
    tiny = np.exp(-700)
    smooth = 0.1 * np.eye(100, M=None, k=0)
    print(all_parater[1]['pai'][1])
    rnk_test=np.zeros(shape=[x_test.shape[0],K])
        # x_test=x_test[y_test[:,k]==1]
    for i in range (x_test.shape[0]):
        for k in range(K):
            miu = all_parater[k]['miu']
            W = all_parater[k]['W']
            pai = all_parater[k]['pai']
            psi = all_parater[k]['psi']
            # print ('miu',k,miu)
            cov = np.matmul(W[k], W[k].T) + psi
            # u, s, v = np.linalg.svd(cov)
            # u = np.where(u < 0, 0, u)
            # s = np.where(s < 0, 0, s)
            # v = np.where(v < 0, 0, v)
            # cov = np.dot(v.T * s, v)
            # if min_eig < 0:
            #     cov -= 10 * min_eig * np.eye(*cov.shape)
            # print(np.linalg.eigvals(cov))i
            cov = np.where(cov < 0, tiny, cov)
            Gaussian=stats.multivariate_normal(x_test[i]-miu[k], cov,allow_singular=True)
            # norm_value=Gaussian.pdf(x_test[i])
            rnk_test[i][k]= pai[k] * Gaussian.pdf(x_test[i])
            #     print ('每一个rnk',rnk_test[i][k][j])
            # print ('第',k,'个模型')
    for i in range(x_test.shape[0]):
        y_pred.append(np.argmax(rnk_test[i]))
        tem=rnk_test[i]
        y_log_prob.append(np.max(tem))
        # print ('aaa',rnk_test[i,:,:])
        ##该图片最大的概率以及label
        # print (dic1)

        # max_label.append(max(dic1))

    # y_pred.append(y_i_pred)

    print( y_pred)
    print (y_log_prob)
    return y_pred,y_log_prob
def make_test_file(x_train,y_train,n):#后n行

    x_test=x_train[:n]
    y_test=y_train[:n]
    np.savez('test.npz', x_test=x_test, y_test=y_test)
    return x_test, y_test

# ## Section 3 `[20%]`: Performance 
# 
# You do not need to do anything in this section but you can use it to test the generalisation performance of your code. We will use it the evaluate the performance of your algorithm on a new test. 

# In[ ]:

def model_performance(x_test, y_test, y_pred, y_log_prob):
    """
    @param x_test: (N,D)-array of features
    @param y_test: (N,C)-array of one-hot-encoded true classes
    @param y_pred: (N,C)-array of one-hot-encoded predicted classes
    @param y_log_prob: (N,C)-array of predicted class log probabilities
    """

    acc = np.all(y_test == y_pred, axis=1).mean()
    llh = y_log_prob[y_test == 1].mean()

    return acc, llh


# In[ ]:

# y_pred, y_log_prob = model_predict(x_test)
# acc, llh = model_performance(x_test, y_test, y_pred, y_log_prob)


# In[ ]:

# 'Average test accuracy=' + str(acc)
#
#
# # In[ ]:
#
# 'Average test likelihood=' + str(llh)




if __name__ == "__main__":
    print ('STart')
    model_train(x_train, y_train)
    x_test, y_test = make_test_file(x_train, y_train, 20)
    model_predict(x_test)
    # acc,llh=model_performance