# Student Name: Dajun Feng		Student ID: z5109061
# Student Name: Xunzhen Long		Student ID: z5049947
# Student Name: Wanqi Tang		Student ID: z5103614

import numpy as np
from scipy import sparse
import os
import gpflow
import tensorflow as tf
from sklearn.decomposition import truncated_svd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
def get_train_file(file,length):
    L=[]
    lines=open(file).readlines()
    for i in range(len(lines)):
        l=[]
        for w in lines[i].split():
            if w!="\n":
                l.append(w)
        l = list(map(int, l))
        L.append(l)
    L=np.array(L)
    P = sparse.lil_matrix((length,2035523))
    for i in range(length):
        s=L[:,0]==i+1
        for j in L[s][:,1]:
            P[i,j]=1
    return P
def get_y_file(file):
    lines = open(file).readlines()
    L=[]
    for i in range(len(lines)):
        for w in lines[i].split():
            if w !='\n':
                w=np.array([w])
                L.append(w)
    L=list(map(int,L))
    L=np.array(L)
    return len(lines),sparse.csr_matrix(L)
def get_all_data(start_file, end_file,step):
    ##-------read file---------##
    file_x = 'conll_train/1.x'
    file_y = 'conll_train/1.y'
    length_sentence, file_y_list = get_y_file(file_y)
    file_x_list = get_train_file(file_x, length_sentence)
    for i in range(start_file,end_file,step):
        file_x='conll_train/'+str(i)+'.x'
        file_y='conll_train/'+str(i)+'.y'
        length_sentence, labels = get_y_file(file_y)
        #####--------stack---------##
        file_y_list=sparse.hstack((file_y_list,labels))
        x_vector = get_train_file(file_x, length_sentence)
        file_x_list = sparse.vstack((file_x_list, x_vector))
    file_y_list=file_y_list.A[0].T
    return file_x_list,file_y_list
def seletc_number_test(file_x_list,file_y_list):
    Selector = SelectKBest(chi2, 3000)
    Selector.fit(file_x_list, file_y_list)
    X_train=Selector.transform(file_x_list)
    reducer=truncated_svd.TruncatedSVD(n_components=500)
    reducer.fit(X_train)
    X_train=reducer.transform(X_train)
    Y_train=np.float16(file_y_list)
    return  X_train,Y_train,reducer,Selector
def read_test_file_and_predice(model,reducer,selector):
    mean_list=[]
    for i in range(8937,10949):
        file_x_name='conll_test_features/'+str(i)+'.x'
        L=[]
        lines=open(file_x_name).readlines()
        for i in range(len(lines)):
            l=[]
            for w in lines[i].split():
                if w!="\n":
                    l.append(w)
            l = list(map(int, l))
            L.append(l)
        L=np.array(L)
        aa=np.max(L, axis=0)
        length=aa[0]
        P = sparse.lil_matrix((length,2035523))
        for i in range(length):
            s=L[:,0]==i+1
            for j in L[s][:,1]:
                P[i,j]=1
        P=selector.transform(P)
        X_test=reducer.transform(P)
        mean,var=model.predict_y(X_test)
        mean_list.append(mean)
    return mean_list
def write_file(mean_list):
    b = os.getcwd() + '\\test_txt\\'
    # print("The created TXT files:")
    if not os.path.exists(b):
        os.makedirs(b)	
    for i in range(len(mean_list)):
        np.savetxt("test_txt/"+str(i)+".txt", mean_list[i],fmt='%.9e',delimiter=",")
    f=open('predictions.txt','w')
    for i in range(len(mean_list)):
        filepath = "test_txt"+'/'+str(i)+".txt"
        for line in open(filepath):
            f.writelines(line)
        f.write('\n')
    f.close()
def test_module(selector,reducer,m):
    # print('Starting Testing.....')
    mnlp = []
    acc = []
    for k in range(10):
        start = k * 800 + 10
        end = k * 800 + 110
        x_valid_1 = 'conll_train/' + str(k + 1) + '.x'
        y_valid_1 = 'conll_train/' + str(k + 1) + '.y'
        length_sentence, file_y_list = get_y_file(y_valid_1)
        file_x_list = get_train_file(x_valid_1, length_sentence)
        for i in range(start, end):
            file_x = 'conll_train/' + str(i) + '.x'
            file_y = 'conll_train/' + str(i) + '.y'
            length_sentence, labels = get_y_file(file_y)
            file_y_list = sparse.hstack((file_y_list, labels))
            x_vector = get_train_file(file_x, length_sentence)
            file_x_list = sparse.vstack((file_x_list, x_vector))
        file_y_list = file_y_list.A[0].T
        X_valid = selector.transform(file_x_list)
        X_SVD = reducer.transform(X_valid)
        X_valid = X_SVD
        Y_valid = file_y_list
        total_org_label_list = []
        mean, var = m.predict_y(X_valid)
        pre_labels = np.argmax(np.log(mean), axis=1)
        mnlp.append(np.average(np.max(np.log(mean), axis=1)))
        for i in range(Y_valid.shape[0]):
            total_org_label_list.append(Y_valid[i])
        total = pre_labels.shape[0]
        correct = 0
        for i in range(pre_labels.shape[0]):
            if pre_labels[i] == total_org_label_list[i]:
                correct += 1
        acc.append(1 - correct / total)
    return np.average(acc),-np.average(mnlp)
def train_module(X_train,Y_train,reducer,selector):
    kenel = gpflow.kernels.RBF(X_train.shape[1], lengthscales=1.0, variance=4.0)
    likel = gpflow.likelihoods.MultiClass(23)
    # print("Training running.........")
    m = gpflow.models.SVGP(
        X_train, np.int8(Y_train), kern=kenel, likelihood=likel, Z=X_train[::3], num_latent=23, whiten=True,
        q_diag=True)
    # print('module built', m)
    opt = gpflow.train.ScipyOptimizer()
    # print('Start optimizing....')
    opt.minimize(model=m, maxiter=30)
    # print('module', m)
    # print('Finish optimizing...')
    return selector,reducer,m

if __name__ == '__main__':
    #-----------------get trainning original data-----------------#
    file_x_list,file_y_list=get_all_data(1,8396,22)
    #-----------------reduce the dimemtion------------------------#
    X_train,Y_train,reducer,selector=seletc_number_test(file_x_list,file_y_list)
    #-----------------training module-----------------------------#
    selector,reducer,m=train_module(X_train,Y_train,reducer,selector)
    #---------------evaluation,get ER and MNLP--------------------#
    er,mnlp=test_module(selector,reducer,m)
    # print("ERROR", er)
    # print('MNLP', mnlp)
    ##------------------ prediction------------------------------##
    mean_list=read_test_file_and_predice(m,reducer,selector)
    for i in range(2012):
        test_label=np.argmax(mean_list[i],axis=1)
        # print('test_label',test_label)
    write_file(mean_list)

    
