"""libs/plot_utils.py
Author: Adam J. Vogt (Aug. 2017)
----------

utils for plotting data and results

"""
import matplotlib.pyplot as plt
import numpy as np

def plot_summary(df):
    """Plot summary of voting record
    Separates percentage of votes by party and vote
    
    Parameters
    ----------
    df : data frame, shape = [438, 17]
        Data frame from voting record csv
        
    Returns
    -------
    None
    
    """
    # Separating votes by party, vote type
    R_tot = df[df['0']=='republican'].shape[0]
    D_tot = df[df['0']=='democrat'].shape[0]
    
    vote_pct = np.asarray([(df[df['0']=='republican'].iloc[:,1:]=='y').sum().values/R_tot,
                    (df[df['0']=='republican'].iloc[:,1:]=='?').sum().values/R_tot,
                    (df[df['0']=='republican'].iloc[:,1:]=='n').sum().values/R_tot,
                    (df[df['0']=='democrat'].iloc[:,1:]=='y').sum().values/D_tot,
                    (df[df['0']=='democrat'].iloc[:,1:]=='?').sum().values/D_tot,
                    (df[df['0']=='democrat'].iloc[:,1:]=='n').sum().values/D_tot])

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,6))
    
    # Republican votes
    ax1.matshow(vote_pct[:3,:], cmap=plt.cm.Reds, 
                alpha=0.3, label='Republican')
    ax1.set_title('Republicans')
    ax1.set_xlabel('Vote')
    ax1.set_ylabel('Type of Vote')
    ax1.xaxis.tick_bottom()
    ax1.set_yticklabels(['','yea','?','nay'])
    ax1.set_xticklabels(['']+[str(i) for i in range(1,17,2)])
    for i in range(vote_pct[:3,:].shape[0]):
        for j in range(vote_pct[:3,:].shape[1]):
            ax1.text(x=j, y=i,
                    s=('%.3f' % vote_pct[:3,:][i, j]),
                    va='center', ha='center',
                    fontsize=9)
    
    # Democrat votes
    ax2.matshow(vote_pct[3:,:], cmap=plt.cm.Blues, 
                alpha=0.3, label='Democrat')
    ax2.set_title('Democrats')
    ax2.set_xlabel('Vote')
    ax2.set_ylabel('Type of Vote')
    ax2.xaxis.tick_bottom()
    ax2.set_yticklabels(['','yea','?','nay'])
    ax2.set_xticklabels(['']+[str(i) for i in range(1,17,2)])
    for i in range(vote_pct[3:,:].shape[0]):
        for j in range(vote_pct[3:,:].shape[1]):
            ax2.text(x=j, y=i,
                    s=('%.3f' % vote_pct[3:,:][i, j]),
                    va='center', ha='center',
                    fontsize=9)
    
    plt.show()


def plot_class_results(X, y, estimator, X_pca, X_lda):
    """Plot results from classification model
    includes PCA Projection scatter,
    LDA Projection scatter, and Confusion Matrix
    
    Parameters
    ----------
    X : array, shape = [438, 48]
        One hot encoded array of votes cast for each measure
    y : array, shape = [438]
        One hot encoded array for party affiliation
        (democrat = 0, republican = 1)
    estimator : estimator object
        trained model for making predictions
    X_pca : array, shape = [438, 48]
        inputs transformed to principal components space
    X_lda : array, shape = [438, 48]
        inputs transformed to linear discriminant analysis space
        
    Returns
    -------
    None
    
    """
    from sklearn.metrics import confusion_matrix
    
    # Making predictions using estimator
    y_pred = estimator.predict(X)

    # Creating PCA & LDA Projection Plots
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.scatter(X_pca[y==1,0], 
                X_pca[y==1,1],
                marker='o', c='red',
                label='Republican')
    ax1.scatter(X_pca[y==0,0], 
                X_pca[y==0,1],
                marker='s', c='blue',
                label='Democrat')
    ax1.scatter(X_pca[y_pred != y,0], 
                X_pca[y_pred != y,1],
                marker='o', facecolors='none',
                edgecolors='black',
                s=100,
                label='Misclassified Samples')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.set_title('Misclassified Samples (PCA)')
    ax1.legend(loc='upper left')
    ax1.set_ylim([-2,3])
    
    # Plotting LDA
    ax2.scatter(X_lda[y==1,0], 
                np.ones([X_lda[y==1].shape[0],1]),
                marker='o', c='red',
                label='Republican')
    ax2.scatter(X_lda[y==0,0], 
                np.zeros([X_lda[y==0].shape[0],1]),
                marker='s', c='blue',
                label='Democrat')
    ax2.scatter(X_lda[(y == 1)&(y_pred != y),0], 
                np.ones([X_lda[(y == 1)&(y_pred != y)].shape[0],1]),
                marker='o', facecolors='none',
                edgecolors='black',
                s=100,
                label='Misclassified Samples')
    ax2.scatter(X_lda[(y == 0)&(y_pred != y),0], 
                np.zeros([X_lda[(y == 0)&(y_pred != y)].shape[0],1]),
                marker='o', facecolors='none',
                edgecolors='black',
                s=100)
    ax2.set_xlabel('Disciminant Variable')
    ax2.set_ylabel('Party Affiliation')
    ax2.set_ylim([-0.05,1.6])
    ax2.tick_params(axis='y',left='off',labelleft='off')
    ax2.set_title('Misclassified Samples (LDA)')
    ax2.legend(loc='upper left')
    plt.show()
    
    #Plotting Confusion Matrix
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,
                    s=cm[i, j],
                    va='center', ha='center')
    ax.set_yticklabels(['','D','R'])
    ax.set_xticklabels(['','D','R'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.show()

