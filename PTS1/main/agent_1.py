import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.base import accuracy_score  # Needed for projection='3d'
from sklearn.manifold import TSNE
from sklearn import preprocessing

import os
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import yaml
os.chdir(os.path.dirname(__file__))
import random
seed = 2934 #2934
random.seed(seed)
np.random.seed(seed)
## todo: normalization before learning

import argparse
import logging
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
def plot_comparison(components, X, y, dim_pref=3, t_sne=False):
    """Draw a scatter plot of points, colored by their labels, before and after applying a learned transformation

    Parameters
    ----------
    components : array_like
        The learned transformation in an array with shape (n_components, n_features).
    X : array_like
        An array of data samples with shape (n_samples, n_features).
    y : array_like
        An array of data labels with shape (n_samples,).
    dim_pref : int
        The preferred number of dimensions to plot (default: 2).
    t_sne : bool
        Whether to use t-SNE to produce the plot or just use the first two dimensions
        of the inputs (default: False).

    """
    if dim_pref < 2 or dim_pref > 3:
        print('Preferred plot dimensionality must be 2 or 3, setting to 2!')
        dim_pref = 2

    if t_sne:
        print("Computing t-SNE embedding")
        tsne = TSNE(n_components=dim_pref, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        Lx_tsne = tsne.fit_transform(X.dot(components.T))
        X = X_tsne
        Lx = Lx_tsne
    else:
        Lx = X.dot(components.T)

    fig = plt.figure(figsize=(2, 1))
    if X.shape[1] > 2 and dim_pref == 3:
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(Lx[:, 0], Lx[:, 1], Lx[:, 2], c=y)
        ax.set_title('Transformed Data')
    elif X.shape[1] == 2:
        ax = fig.add_subplot(121)
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.set_title('Original Data')
        ax = fig.add_subplot(122)
        ax.scatter(Lx[:, 0], Lx[:, 1], c=y)
        ax.set_title('Transformed Data')
        
    
    plt.savefig(cfg[dataset_name]['fig_save_path']+'3dvisualCompare.png')
    plt.show()


def read_rest_data(fpath):
    
    df = pd.read_csv(fpath)
    
    # names_feature = ['Pos 1', 'Pos 2', 'Pos 3', 'score', 'target', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3', 'C3',
    # 'Final_intermolecular_energy', 'vdW_Hbond_desolv', 'Electrostatic_Energy', 'Final_Total_Internal_Energy', 'kd']
    used_feature = ['A', 'B', 'C']
    
    x = np.array(df[used_feature])
    names = np.array(df[['resn']])
    
    # standard_scaler = preprocessing.StandardScaler()
    # x = standard_scaler.fit_transform(x)
    
    return x, names
    
def read_dataset(fpath, shuffle=False, augment=False):
    df = pd.read_csv(fpath)
    
    # names_feature = ['Pos 1', 'Pos 2', 'Pos 3', 'score', 'target', 'A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'A3', 'B3', 'C3',
    # 'Final_intermolecular_energy', 'vdW_Hbond_desolv', 'Electrostatic_Energy', 'Final_Total_Internal_Energy', 'kd']
    used_feature = ['Pos 1', 'Pos 2', 'Pos 3']
    
    xy = df.loc[df['target'] != 'unknown'][:]
    
    x = np.array(xy[used_feature])
    y = np.array([1 if v=='good' else 0 for v in xy['target']])
    
    if shuffle:
        perm_idx = np.arange(len(y))
        np.random.shuffle(perm_idx)
        train_ind = perm_idx[0: int(len(y)*0.9)]
        val_ind = perm_idx[int(len(y)*0.9):]
    
        train_X, train_Y = x[train_ind], y[train_ind]
        val_X, val_Y = x[val_ind], y[val_ind]
    
    else:
        print('empty val_X')
        train_X, train_Y = x,y
        val_X, val_Y = [], []
    
    
    
    if augment:
        x_0 = train_X[train_Y==0]
        
        N = 400
        new_data = list()
        for i in range(N):
            ind_a = np.random.choice(len(x_0))
            ind_b = np.random.choice(len(x_0))
            w = np.random.uniform(0,1)
            c = w*x_0[ind_a] + (1-w)*x_0[ind_b]
            new_data.append(c)
        
        train_X = np.concatenate((train_X, new_data))
        
        new_y = np.array([0 for i in range(N)])
        train_Y = np.concatenate((train_Y, new_y))
        
    
    
    test_x = np.array(df.loc[df['target'] == 'unknown'][used_feature])
    test_name = np.array(df.loc[df['target'] == 'unknown'][['resn']])
    
    if False:
        more_budingwei = pd.read_csv('../exp/more_data/new_budingwei_test.csv')
        more_dingwei = pd.read_csv('../exp/more_data/new_dingwei_test.csv')
        x_more = np.concatenate((np.array(more_dingwei[['A', 'B', 'C']]), np.array(more_budingwei[['A', 'B', 'C']])), axis=0 )
        y_more = np.concatenate((np.array([1 if v=='good' else 0 for v in more_dingwei['target']]), np.array(np.array([1 if v=='good' else 0 for v in more_budingwei['target']]))),  axis=0)
        val_X = x_more
        val_Y = y_more
    else:
        more_budingwei = pd.read_csv('../exp/more_data/new_budingwei_test.csv')
        more_dingwei = pd.read_csv('../exp/more_data/new_dingwei_test.csv')
        x_more = np.concatenate((np.array(more_dingwei[['A', 'B', 'C']]), np.array(more_budingwei[['A', 'B', 'C']])), axis=0 )
        y_more = np.concatenate((np.array([1 if v=='good' else 0 for v in more_dingwei['target']]), np.array(np.array([1 if v=='good' else 0 for v in more_budingwei['target']]))),  axis=0)
        val_X = x_more
        val_Y = y_more
    
    return train_X, train_Y, val_X, val_Y, test_x, test_name

def load_dataset(dataset_name, cfg):
    if dataset_name == 'PTS1':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_PTS1(cfg)
    elif dataset_name == 'GB1':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1(cfg)
    elif dataset_name == 'PhoQ':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_PhoQ(cfg)
    elif dataset_name == 'GB1_score3_consider1ps':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1_score3_consider1ps(cfg)
    elif dataset_name == 'GB1_set1ps_change2ps_consider1ps':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1_set1ps_change2ps_consider1ps(cfg)
    elif dataset_name == 'GB1_syn':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_GB1_syn(cfg)
    elif dataset_name == '2d_syn':
        train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y = read_dataset_2d_syn(cfg)
    
    used_features = []
    return train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_Y
    
def read_dataset_2d_syn(cfg):
    # 按照数据集名字读取数据
    n_f = 2
    GB1 = np.load(cfg['2d_syn']['processed_data_path']+'Pre_data.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
    
    augment = False
    if augment:
        x_0 = train_X[train_Y==1]
        
        N = 500
        new_data = list()
        for i in range(N):
            ind_a = np.random.choice(len(x_0))
            ind_b = np.random.choice(len(x_0))
            w = np.random.uniform(0,1)
            c = w*x_0[ind_a] + (1-w)*x_0[ind_b]
            new_data.append(c)
        
        train_X = np.concatenate((train_X, new_data))
        
        new_y = np.array([1 for i in range(N)])
        train_Y = np.concatenate((train_Y, new_y))
        
    
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,

def read_dataset_GB1_syn(cfg):
    n_f = 4
    GB1 = np.load(cfg['GB1_syn']['processed_data_path']+'Pre_data.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
    
    augment = True
    if augment:
        x_0 = train_X[train_Y==1]
        
        N = 500
        new_data = list()
        for i in range(N):
            ind_a = np.random.choice(len(x_0))
            ind_b = np.random.choice(len(x_0))
            w = np.random.uniform(0,1)
            c = w*x_0[ind_a] + (1-w)*x_0[ind_b]
            new_data.append(c)
        
        train_X = np.concatenate((train_X, new_data))
        
        new_y = np.array([1 for i in range(N)])
        train_Y = np.concatenate((train_Y, new_y))
        
    
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,
    
def read_dataset_PTS1(cfg):
    fpath = cfg['PTS1']['dataset_path']
    rest_fpath = cfg['PTS1']['testset_path']
    train_X, train_Y, val_X,  val_Y, test_x, test_name = read_dataset(fpath, shuffle=True, augment=False)
    
    rest_data, rest_names = read_rest_data(rest_fpath)
    rest_label = np.ones_like(rest_names)
    print('*'*20+'fake test_label!')
    
    return train_X, train_Y, val_X,  val_Y, test_x, test_name, rest_data, rest_names, rest_label


def read_dataset_GB1(cfg):
    # * input: read p1, p2, p3, p4, f1,f2,f3, f4
    # * output 
    
    n_f = 4
    GB1 = np.load(cfg['GB1']['processed_data_path']+'Pre_data.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,

def read_dataset_PhoQ(cfg):
    n_f = 4
    GB1 = np.load(cfg['PhoQ']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,
    

def read_dataset_GB1_score3_consider1ps(cfg):
    n_f = 4
   
    GB1 = np.load(cfg['GB1_score3_consider1ps']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,

def read_dataset_GB1_set1ps_change2ps_consider1ps(cfg):
    n_f = 4
   
    GB1 = np.load(cfg['GB1_set1ps_change2ps_consider1ps']['processed_data_path']+'Pre_PhoQ.npz', allow_pickle=True)
    train_X, train_Y = GB1['train_X'], GB1['train_Y']
    val_X, val_Y = GB1['val_X'], GB1['val_Y']
    if len(val_X) == 0:
        val_X, val_Y = train_X, train_Y
    
    test_X, test_Y = GB1['test_X'], GB1['test_Y']
    test_names = GB1['test_names']
           
    return train_X[:, 0:n_f], train_Y, val_X[:, 0:n_f], val_Y, val_X[:, 0:n_f], val_Y, test_X[:, 0:n_f], test_names, test_Y,



class LDA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Within class scatter matrix:
        # SW = sum((X_c - mean_X_c)^2 )

        # Between class scatter:
        # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            SW += (X_c - mean_c).T.dot((X_c - mean_c))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)

        self.SW = SW
        # Determine SW^-1 * SB
        A = np.linalg.inv(SW).dot(SB)
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        eigenvalues, eigenvectors = np.linalg.eig(A)
        # -> eigenvector v = [:,i] column vector, transpose for easier calculations
        # sort eigenvalues high to low
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # store first n eigenvectors
        self.linear_discriminants = np.array(eigenvectors[0:self.n_components], dtype=np.float32)

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)

    def predict(self, X, mu):
        x = self.transform(X)
        return x>mu

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_data_to_plot(trainX,trainY):
    
    # K折交叉验证,这里设置K=5

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(trainX)

    fold = 0

    epoch_list = []
        
    for train_index, test_index in kf.split(trainX):
        fold += 1
        print(f'fold {fold}')

        fold_data = {}

        for epoch in range(100):

            train, test = trainX[train_index], trainX[test_index]
            train_label, test_label = trainY[train_index], trainY[test_index]
            #train = trainX
            #train_label = trainY


            dimen = 1
            lda = LDA(dimen)
            
            lda.fit(train, train_label)


            lr = LogisticRegression().fit(lda.transform(train), train_label)
            test_probs = lr.predict_proba(lda.transform(test)) 

            

            fold_data[f'epoch{epoch}fold{fold}'] =  {
                'y_pred': test_probs,
                'y_test': test_label
            } 

            epoch_list.append(fold_data) 
        


    np.savez('../score/PTS1_agent1.npz', data=epoch_list)

def decision(x,label,midpoint1):
    num_correct=[0,0]
                
    X_c = np.array(x[label == 0], dtype = np.float32)
    num_correct[0] += np.sum((X_c < midpoint1)) 
    X_c = np.array(x[label == 1], dtype = np.float32)
    num_correct[1] += np.sum((X_c > midpoint1)) 
                
    recall0 = num_correct[0]/len(x[label == 0])
    recall1 = num_correct[1]/len(x[label == 1])
    recall = np.sum(num_correct)/len(label)
                
    acc0 = num_correct[0]/len(x < midpoint1)
    acc1 = num_correct[1]/len(x > midpoint1)
    accuracy = num_correct[1]/np.sum(x>midpoint1)

    return recall0, recall1, recall, acc0, acc1, accuracy

def get_data_to_plot2(trainX,trainY):
    
    # K折交叉验证,这里设置K=5
 
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(trainX)

    fold = 0
        
    for train_index, test_index in kf.split(trainX):
        fold += 1
        print(f'fold {fold}')

        fold_data = {}
        
        for epoch in range(1):
        
            train, test = trainX[train_index], trainX[test_index]
            train_label, test_label = trainY[train_index], trainY[test_index]
            #train = trainX
            #train_label = trainY
        
            
            print(f'good: {sum(train_label==1)}')
            print(f'bad: {sum(train_label==0)}')

            dimen = 1
            lda = LDA(dimen)
            
            lda.fit(train, train_label)
            if dimen > 1:
                plot_comparison(lda.linear_discriminants, train, train_label, dim_pref=dimen)
            else:
                                
                x = lda.transform(train)
                mu1 = []
                X_c = x[train_label == 0]
                mu1.append(np.mean(X_c, axis=0))
                
                X_c = x[train_label == 1]
                mu1.append(np.mean(X_c, axis=0)) 
                
                midpoint1 = (mu1[1]+mu1[0])/2
                
                #plt.scatter(x, np.zeros(len(x)), c=train_label)
                db_y = np.arange(-0.5, 0.5, 0.01)

                #plt.plot(midpoint1*np.ones(len(db_y)), db_y, '--b')
                
                
                num_correct=[0,0]
                
                X_c = np.array(x[train_label == 0], dtype = np.float32)
                num_correct[0] += np.sum((X_c < midpoint1)) 
                X_c = np.array(x[train_label == 1], dtype = np.float32)
                num_correct[1] += np.sum((X_c > midpoint1)) 
                
                recall0 = num_correct[0]/len(x[train_label == 0])
                recall1 = num_correct[1]/len(x[train_label == 1])
                recall = np.sum(num_correct)/len(train_label)
                
                acc0 = num_correct[0]/len(x < midpoint1)
                acc1 = num_correct[1]/len(x > midpoint1)
                accuracy = num_correct[1]/np.sum(x>midpoint1)
                
                
                #print(f'num examples: {len(train_label)}')
                #print(f'Best Parameter: {lda.linear_discriminants}, Decision Boundary: mu={midpoint1}')
                #print(f"Train acc budingwei: {acc0:.2%}, acc dingwei: {acc1:.2%}, overall acc: {accuracy:.2%}")
                #print(f'Train recall budingwei: {recall0:.2%}, recall dingwei: {recall1:.2%}, overall recall: {recall:.2%}')
                #print(f'accuracy: {accuracy}')
                
            num_correct=[0,0]
            test_X_ = lda.transform(test)
            test_recall0, test_recall1, test_recall, test_acc0, test_acc1, test_accuracy = decision(
                x=test_X_,
                label=test_label,
                midpoint1=midpoint1
            )
            
            print(f'Best Parameter: {lda.linear_discriminants}, Decision Boundary: mu={midpoint1}')
            print(f"Test acc budingwei: {test_acc0:.2%}, acc dingwei: {test_acc1:.2%}, overall acc: {test_accuracy:.2%}")
            print(f'Test recall budingwei: {test_recall0:.2%}, recall dingwei: {test_recall1:.2%}, overall recall: {test_recall:.2%}')
            
            test_predict = test_X_ > midpoint1
            print('-'*20 + ' result ' + '-'*20)
            print(f'recall score: {recall_score(test_label, test_predict)}')
            print(f'precision score: {precision_score(test_label, test_predict)}')
            print(f'auc score: {roc_auc_score(test_label, test_predict)}')
            print(f'classification report: {classification_report(test_label, test_predict)}')
            
        # X_c = np.array(test_X_[test_label == 0], dtype = np.float32)
        # num_correct[0] += np.sum((X_c < midpoint1)) 
        # X_c = np.array(test_X_[test_label == 1], dtype = np.float32)
        # num_correct[1] += np.sum((X_c > midpoint1)) 
        
        # test_recall0 = num_correct[0]/len(test_X_[test_label == 0])
        # test_recall1 = num_correct[1]/len(test_X_[test_label == 1])
        # test_recall = np.sum(num_correct)/len(test_label)
        
        # test_acc0 = num_correct[0]/np.sum(test_X_ < midpoint1)
        # test_acc1 = num_correct[1]/np.sum(test_X_ > midpoint1)
        # test_acc = np.sum(num_correct)/len(test_label)
        
        # print(f"test recall0: {test_recall0:.2%}, recall1: {test_recall1:.2%}, overall recall: {test_recall:.2%}")
        # print(f"test acc0: {test_acc0:.2%}, acc1: {test_acc1:.2%}, overall acc: {test_acc:.2%}")

        
    # plt.show()
    # plt.savefig(cfg[dataset_name]['fig_save_path']+'visualCompare.png')   

def lda_exp(dataset_name):
    
    print(f'good: {sum(train_Y==1)}')
    print(f'bad: {sum(train_Y==0)}')

    

    if False:
        get_data_to_plot(train_X, train_Y)
    if True:
        get_data_to_plot2(train_X, train_Y)

    dimen = 1
    lda = LDA(dimen)
    lda.fit(train_X, train_Y)
    if dimen > 1:
        plot_comparison(lda.linear_discriminants, train_X, train_Y, dim_pref=dimen)
       
    else:
        plt.figure(figsize=(3, 2))
        x = lda.transform(train_X)
        mu1 = []
        X_c = x[train_Y == 0]
        mu1.append(np.mean(X_c, axis=0))
        
        X_c = x[train_Y == 1]
        mu1.append(np.mean(X_c, axis=0)) 
        
        midpoint1 = (mu1[1]+mu1[0])/2
        
        plt.scatter(x, np.zeros(len(x)), c=train_Y).figure.set_dpi(720)
        db_y = np.arange(-0.5, 0.5, 0.01)

        plt.plot(midpoint1*np.ones(len(db_y)), db_y, '--b')
        
        
        num_correct=[0,0]
        
        X_c = np.array(x[train_Y == 0], dtype = np.float32)
        num_correct[0] += np.sum((X_c < midpoint1)) 
        X_c = np.array(x[train_Y == 1], dtype = np.float32)
        num_correct[1] += np.sum((X_c > midpoint1)) 
        
        acc0 = num_correct[0]/len(x[train_Y == 0])
        acc1 = num_correct[1]/len(x[train_Y == 1])
        acc = np.sum(num_correct)/len(train_Y)

        accuracy = num_correct[1]/np.sum(x>midpoint1)
        
        
        print(f'num examples: {len(train_Y)}')
        print(f'Best Parameter: {lda.linear_discriminants}, Decision Boundary: mu={midpoint1}')
        # print(f"Train acc0: {acc0:.2%}, acc1: {acc1:.2%}, overall acc: {acc:.2%}")
        
        predict = x > midpoint1
        print(f'Train: accuracy: {accuracy_score(train_Y, predict)}, recall score: {recall_score(train_Y, predict)}, precision: {precision_score(train_Y, predict)}')
        print(f'Train report: \n {classification_report(train_Y, predict)}')
        
    num_correct=[0,0]
    val_X_ = lda.transform(val_X)
    X_c = np.array(val_X_[val_Y == 0], dtype = np.float32)
    num_correct[0] += np.sum((X_c < midpoint1)) 
    X_c = np.array(val_X_[val_Y == 1], dtype = np.float32)
    num_correct[1] += np.sum((X_c > midpoint1)) 
    
    val_acc0 = num_correct[0]/len(val_X_[val_Y == 0])
    val_acc1 = num_correct[1]/len(val_X_[val_Y == 1])
    val_acc = np.sum(num_correct)/len(val_Y)
    # print(f"Validation acc0: {val_acc0:.2%}, acc1: {val_acc1:.2%}, overall acc: {val_acc:.2%}")
    val_predict = val_X_ > midpoint1
    print(f"Validation: accuracy: {accuracy_score(val_Y, val_predict)}, recall: {recall_score(val_Y, val_predict)}, precision: {precision_score(val_Y, val_predict)}")
    print(f"Validation report: \n {classification_report(val_Y, val_predict)}")

    prediction = lda.predict(test_x, midpoint1)
    ret = zip(test_name, prediction)
    for i in range(len(test_name)):
        print(f'name: {test_name[i]}, prediction: {prediction[i]}')
    
    
    rest_prediction = lda.predict(rest_data, midpoint1)
    rest_map = dict(zip(list(np.squeeze(rest_names)), list(np.squeeze(rest_prediction))))
    rest_dict = sorted(rest_map.items(), key = lambda d: d[1], reverse=True)
    
   
            
    rest_pd = pd.DataFrame(data=rest_dict, columns=['resn', 'prediction'])
    rest_pd.to_csv(cfg[dataset_name]['result_save_path'] + 'rest_prediction_pos.csv')
    print(f'Finished saving prediction for unknown data, number of positive predictions {np.sum(np.squeeze(rest_prediction))}')
    path = cfg[dataset_name]['result_save_path'] + 'rest_prediction_pos.csv'
    print(f'save path: {path}')
 
    
    plt.savefig(cfg[dataset_name]['fig_save_path']+'visualCompare.png', bbox_inches='tight')
    plt.show()
    

def lda_cv_exp(dataset_name, trainX, trainY):
    
    for amount in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(trainX)
        
        for train_index, test_index in kf.split(trainX):
            train, test = trainX[train_index], trainX[test_index]
            train_label, test_label = trainY[train_index], trainY[test_index]
            train = trainX
            train_label = trainY
        
            
            print(f'good: {sum(train_label==1)}')
            print(f'bad: {sum(train_label==0)}')

            dimen = 1
            lda = LDA(dimen)
            
            lda.fit(train, train_label)
            if dimen > 1:
                plot_comparison(lda.linear_discriminants, train, train_label, dim_pref=dimen)
            else:
                x = lda.transform(train)
                mu1 = []
                X_c = x[train_label == 0]
                mu1.append(np.mean(X_c, axis=0))
                
                X_c = x[train_label == 1]
                mu1.append(np.mean(X_c, axis=0)) 
                
                midpoint1 = (mu1[1]+mu1[0])/2
                
                plt.scatter(x, np.zeros(len(x)), c=train_label)
                db_y = np.arange(-0.5, 0.5, 0.01)

                plt.plot(midpoint1*np.ones(len(db_y)), db_y, '--b')
                
                
                num_correct=[0,0]
                
                X_c = np.array(x[train_label == 0], dtype = np.float32)
                num_correct[0] += np.sum((X_c < midpoint1)) 
                X_c = np.array(x[train_label == 1], dtype = np.float32)
                num_correct[1] += np.sum((X_c > midpoint1)) 
                
                recall0 = num_correct[0]/len(x[train_label == 0])
                recall1 = num_correct[1]/len(x[train_label == 1])
                recall = np.sum(num_correct)/len(train_label)
                
                acc0 = num_correct[0]/len(x < midpoint1)
                acc1 = num_correct[1]/len(x > midpoint1)
                accuracy = num_correct[1]/np.sum(x>midpoint1)
                
                
                print(f'num examples: {len(train_label)}')
                print(f'Best Parameter: {lda.linear_discriminants}, Decision Boundary: mu={midpoint1}')
                print(f"Train acc0: {acc0:.2%}, acc1: {acc1:.2%}, overall acc: {accuracy:.2%}")
                print(f'Train recall0: {recall0:.2%}, recall1: {recall1:.2%}, overall recall: {recall:.2%}')
                print(f'accuracy: {accuracy}')
                
            num_correct=[0,0]
            test_X_ = lda.transform(test)
            test_predict = test_X_ > midpoint1
            print('-'*20 + ' result ' + '-'*20)
            print(f'recall score: {recall_score(test_label, test_predict)}')
            print(f'precision score: {precision_score(test_label, test_predict)}')
            print(f'auc score: {roc_auc_score(test_label, test_predict)}')
            print(f'classification report: {classification_report(test_label, test_predict)}')
            
        # X_c = np.array(test_X_[test_label == 0], dtype = np.float32)
        # num_correct[0] += np.sum((X_c < midpoint1)) 
        # X_c = np.array(test_X_[test_label == 1], dtype = np.float32)
        # num_correct[1] += np.sum((X_c > midpoint1)) 
        
        # test_recall0 = num_correct[0]/len(test_X_[test_label == 0])
        # test_recall1 = num_correct[1]/len(test_X_[test_label == 1])
        # test_recall = np.sum(num_correct)/len(test_label)
        
        # test_acc0 = num_correct[0]/np.sum(test_X_ < midpoint1)
        # test_acc1 = num_correct[1]/np.sum(test_X_ > midpoint1)
        # test_acc = np.sum(num_correct)/len(test_label)
        
        # print(f"test recall0: {test_recall0:.2%}, recall1: {test_recall1:.2%}, overall recall: {test_recall:.2%}")
        # print(f"test acc0: {test_acc0:.2%}, acc1: {test_acc1:.2%}, overall acc: {test_acc:.2%}")

        
    # plt.show()
    # plt.savefig(cfg[dataset_name]['fig_save_path']+'visualCompare.png')

def compare_rank(y_true, y_pred, name):
    true_curve = sorted(np.squeeze(y_true))
    index = np.argsort(np.squeeze(y_true))
    tmp2 = list(np.squeeze(y_pred))
    pred_curve = [tmp2[i] for i in index]
    
    
    plt.plot(true_curve, 'ob', label = 'True label')
    plt.plot(pred_curve, 'xr', label = 'Predicted label')
    plt.legend()
    plt.xlabel('rank')
    plt.ylabel('value')
    plt.title(cfg[dataset_name]['fig_save_path']+name + '_true_vs_pred')
    plt.savefig(cfg[dataset_name]['fig_save_path']+name + '_true_vs_pred.png')
    plt.show()
    plt.close()
       
    
def run_classifier(dataset_name, negative_num=0):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import tree
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.svm import OneClassSVM
    import sklearn.metrics as metrics
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve, roc_auc_score
    import seaborn as sns
    
    # names = [ 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM', 'one_class_SVM']
    # classfiers = [
    #     # MultinomialNB(alpha=0.01),
    #     KNeighborsClassifier(),
    #     LogisticRegression(penalty='l2'),
    #     RandomForestClassifier(n_estimators=8),
    #     tree.DecisionTreeClassifier(),
    #     GradientBoostingClassifier(n_estimators=200),
    #     SVC(kernel='rbf', probability=True),
    #     OneClassSVM(gamma='scale')
    # ]
    # def try_different_method(tmp_name, model):
    #     if tmp_name == 'one_class_SVM':
    #         model.fit(train_X)
    #         score = 1
    #     else:
    #         model.fit(train_X,train_Y)
    #         score = model.score(val_X, val_Y) # score为拟合优度，越大，说明x对y的解释程度越高
    #     result = model.predict(val_X)
    #     #plt.figure()
    #     plt.plot(np.arange(len(result)), val_Y,'g-',label='true value')
    #     plt.plot(np.arange(len(result)),result,'r-',label='predict value')
    #     plt.title('%s score: %f' % (tmp_name,score))
    #     plt.legend()
    
    # plt.figure(figsize=(20, 20))
    # # sns.set_style("white")
    # for i in range(0,len(names)):
    #     ax = plt.subplot(3,3,i+1)
    #     plt.xlim(0,20) # 这里只选择绘图展示前20个数据的拟合效果，但score是全部验证数据的得分
    #     try_different_method(names[i],classfiers[i])
    # plt.show()
    # plt.savefig(cfg[dataset_name]['fig_save_path'] + 'compare_classifier.png')
    # plt.close()
    
    # # ! gred search
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    # 构造基于GridSearchCV的结果打印功能
    class grid():
        def __init__(self,model):
            self.model = model

        def grid_get(self,X,y,param_grid):
            grid_search = GridSearchCV(self.model,param_grid,cv=5,scoring='recall')#, scoring=rmsle_scorer, n_jobs=-1) #n_jobs调用所有核并行计算
            grid_search.fit(X,y)
            #这里为了方便比较mean_test_score，同一取负值并开方
            print('Best params is ',grid_search.best_params_, grid_search.best_score_) #打印最优参数
            grid_search.cv_results_['mean_test_score'] = grid_search.cv_results_['mean_test_score']
            pd.set_option('display.max_columns', None) # 显示所有列
            pd.set_option('max_colwidth',100) # 设置列的显示长度为100，默认为50
            print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']]) #打印所有结果

    #以随机森林为例进行调参，可更改为其他方法，变更grid()内函数即可
    #这里的np.log1p(y_train)是为了保证y_train值尽量贴近正态分布，利于回归拟合，请根据实际情况进行调整
    #同时，np.log1p(y_train)的处理也是为了和rmsle评估函数的功能对应起来，因为该函数默认会先进行log变换回去（convertExp=True）
    
    # grid(GradientBoostingClassifier()).grid_get(train_X, train_Y,{'random_state':[42], 'n_estimators':[100, 120, 140, 160, 180, 200, 220, 240, 260]}) #随机森林
    # grid(OneClassSVM()).grid_get(train_X, train_Y, { 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}) #随机森林
    
    # import pdb; pdb.set_trace()
    
    
    # * chose one classifier
    regr = GradientBoostingClassifier(n_estimators=80)
    # regr =  SVC(kernel='rbf', probability=True, class_weight='balanced')
    # regr = LogisticRegression(penalty='l2')
    regr.fit(train_X, train_Y)
    # regr = OneClassSVM(gamma='scale', kernel='rbf')
    # regr.fit(train_X[train_Y==1, :])
    
   
    # The mean squared error
    target_names = ['budingwei', 'dingwei']
    support = classification_report(train_Y, regr.predict(train_X), target_names=target_names)
    print(support)
    
    accuracy = accuracy_score(train_Y, regr.predict(train_X))
    print("Train accuracy:", accuracy)
    compare_rank(train_Y, regr.predict(train_X), name='GB_train')
    print("="*30)
    
    # support = classification_report(val_Y, regr.predict(val_X), target_names=target_names)
    # print(support)
    # accuracy = accuracy_score(val_Y, regr.predict(val_X))
    # print("Val accuracy:", accuracy)
    # compare_rank(val_Y, regr.predict(val_X), name='GB_val')
    # print("="*30)
    predict_rest = regr.predict(rest_data)
    support = classification_report(rest_label, predict_rest, target_names=target_names)
    print(support)
    logger.info(support)
    
    precision, recall = precision_score(rest_label, predict_rest), recall_score(rest_label, predict_rest)
    np.savez(cfg[dataset_name]['fig_save_path']+'GB1_precision_recall_agent1_'+str(negative_num)+'.npz', **{'precision': precision, 'recall': recall})
    
    accuracy = accuracy_score(rest_label, predict_rest)
    print(f"Test accuracy: {accuracy}")
    compare_rank(rest_label, regr.predict(rest_data), name='GB_test')
    
    
    # filter test
    test_predict = regr.predict(rest_data)  
    print(f'Classified Positive: {np.sum(np.array(test_predict)==1)}')
    logger.info(f'Classified Positive: {np.sum(np.array(test_predict)==1)}')
    plt.figure()
    plt.scatter(range(len(test_predict)), test_predict,label='true value')
    plt.scatter(range(len(rest_label)), rest_label,label='predict value')
    plt.legend()
    
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'agent1_pred_cls.png')
    plt.show()
    plt.close()

    
    # * draw roc curve
    test_predict_prob = regr.predict_proba(rest_data)

    lr_auc= roc_auc_score(rest_label, test_predict_prob[:,1])
    GB1_roc = {'test_Y': rest_label, 'test_predict_prob': test_predict_prob, 'lr_auc':lr_auc}
    np.savez(cfg[dataset_name]['fig_save_path']+'GB1_roc_agent1_'+str(negative_num)+'.npz', **GB1_roc)
    
    
    
    # sort by score
    predict_rank = dict(zip(np.squeeze(rest_names), test_predict))
    sort_predict_rank = sorted(predict_rank.items(), key=lambda d: d[1], reverse=True)
    rest_pd = pd.DataFrame(data=sort_predict_rank, columns=['resn', 'prediction'])
    rest_pd.to_csv(cfg[dataset_name]['result_save_path'] + 'agent1_classification_top1000.csv')
    
    top_freq =np.array([sort_predict_rank[i][1] for i in range(len(sort_predict_rank))])
    # length = min(int(sum(top_freq[top_freq==1])), 1000)
    # top_sort_predict_rank = [sort_predict_rank[i][0] for i in range(length)] # rank top 1000 
    # top_sort_predict_rank_save = pd.DataFrame(top_sort_predict_rank, columns=['resn'])
    # top_sort_predict_rank_save.to_csv(cfg[dataset_name]['result_save_path']+'agent1_classification_top1000.csv')
    

    tmp_freq = np.exp(top_freq)-1
    plt.plot(top_freq)
    
    plt.savefig(cfg[dataset_name]['fig_save_path'] + 'agent1_pred_cls.png')
    plt.show()
        
        
  
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='data amount influence')
    parser.add_argument('--log_file', dest='log_file', type=str, help='log file', default='data.log')
    parser.add_argument('--negative_num', dest='negative_num', type=int, help='additional negative_num', default=2000)
    args = parser.parse_args()
    
    logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)
    logger.info(args.log_file)
    
    
    with open('agent1_config.yaml', 'r') as config:
        cfg = yaml.safe_load(config)
    dataset_name = 'PTS1' #'GB1_set1ps_change2ps_consider1ps'
    # 'GB1_score3_consider1ps'
    # fpath = cfg[dataset_name]['dataset_path']
    # rest_fpath = cfg[dataset_name]['testset_path']
    # train_X, train_Y, val_X,  val_Y, test_x, test_name= read_dataset(fpath, shuffle=True, augment=True)
    # rest_data, rest_names = read_rest_data(rest_fpath)
    
    train_X, train_Y, val_X, val_Y, test_x, test_name, rest_data, rest_names, rest_label = load_dataset(dataset_name, cfg)
    
    # * save kFold cv dataset
    # kf = KFold(n_splits=5, shuffle=True)
    # kf.get_n_splits(train_X)
    
    # for index, (train_index, test_index) in enumerate(kf.split(train_X)):
    #     train, test = train_X[train_index], train_X[test_index]
    #     train_label, test_label = train_Y[train_index], train_Y[test_index]
    #     pts1_cv = {'train_X': train, 'train_Y':train_label, 'test_X': test, 'test_Y': test_label}

    #     np.savez('/data/gzc/code/lang_dataset/lang_datasets_new/data/pts1_cvfig/cv_'+str(index)+'_amount_1'+'.npz', **pts1_cv)
    #* section end 
    
    if dataset_name != 'PTS1':
        train_Y[train_Y==0]=-1
        val_Y[val_Y==0]=-1  
      
    if type(train_Y) != list:
        train_Y= train_Y.astype('int')
    if type(val_Y) != list:
        val_Y=val_Y.astype('int')
    if type(rest_label) != list:
        rest_label=rest_label.astype('int')
    
    if len(test_x)==0:
        test_x = val_X
    tmp_x = np.concatenate((train_X, val_X))
    tmp_x = np.concatenate((tmp_x, test_x))
    tmp_x = np.concatenate((tmp_x, rest_data))

    standard_scaler = preprocessing.StandardScaler()
    tmp_x = standard_scaler.fit_transform(tmp_x)
    train_X = standard_scaler.transform(train_X)
    val_X = standard_scaler.transform(val_X)
    test_x = standard_scaler.transform(test_x)
    rest_data = standard_scaler.transform(rest_data)

    # 
    if dataset_name == 'PTS1':
        lda_exp(dataset_name)
        #lda_cv_exp(dataset_name, trainX=train_X, trainY=train_Y)
    else:
        run_classifier(dataset_name, args.negative_num)
        







