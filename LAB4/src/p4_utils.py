import numpy as np
import numpy.matlib as matl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def createDataSet(n,model,ymargin):
    x = np.random.rand(n,2)*2.0*np.pi

    if model == 'sine':
        c = x[:,1] > np.sin(x[:,0])
    elif model == 'linear':
        c = x[:,1] > x[:,0]
    elif model == 'square':
        c = x[:,1] > x[:,0]*x[:,0]
    else:
        c = x[:,1] > x[:,0]
    
    x[c == True,1] += ymargin
    x[c == False,1] -= ymargin

    return x, 2*c-1

def plotData(x,y,c,style0,style1,title):
    plt.plot(x[c==-1],y[c==-1],style0)
    plt.plot(x[c==1],y[c==1],style1)
    plt.grid(True)
    plt.axis([x.min()-0.2, x.max()+0.2, y.min()-0.2, y.max()+0.2])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)

def plotModel(x,y,clase,clf,title):
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2
    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
    plt.contour(xx, yy, z, [-1, 0, 1], linewidths=[2, 2, 2], colors=['#FF0000', 'k', '#0000FF'])

    plt.scatter(x[clase==-1], y[clase==-1], c='#FF0000')
    plt.scatter(x[clase==1], y[clase==1], c='#0000FF')
    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
