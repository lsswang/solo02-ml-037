"""Existing local tanh classifier, trained from exact local labels."""
import numpy as np
from scipy.special import logsumexp

def forward(model,X):
    A,b,C,c=[np.asarray(model[k],dtype=float) for k in ['A','b','C','c']]
    H=np.tanh(np.asarray(X)@A+b);Z=H@C+c;P=np.exp(Z-logsumexp(Z,axis=1,keepdims=True))
    return H,Z,P

def objective(model,X,labels,weights,reg):
    X=np.asarray(X);y=np.asarray(labels);a=np.asarray(weights);H,Z,P=forward(model,X);total=a.sum()
    loss=float(np.sum(a*(logsumexp(Z,axis=1)-Z[np.arange(len(y)),y]))/total)
    G=P.copy();G[np.arange(len(y)),y]-=1;G*=a[:,None]/total
    C=np.asarray(model['C']);dH=(G@C.T)*(1-H*H)
    grads={'A':X.T@dH,'b':dH.sum(axis=0),'C':H.T@G,'c':G.sum(axis=0)}
    for k in grads:
        v=np.asarray(model[k]);loss+=reg/2*float(np.sum(v*v));grads[k]+=reg*v
    return loss,grads
