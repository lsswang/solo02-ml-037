"""Existing zero-padded mean, valid when every client has all labels."""
import numpy as np

def aggregate(reports,classes):
    R=len(reports[0]['probabilities']);out=np.zeros((R,classes));mass=np.zeros(R)
    for report in reports:
        ids=report['labels'];p=np.asarray(report['probabilities']);weight=report['weight']
        for r,present in enumerate(report['present']):
            if present:out[r,ids]+=weight*p[r];mass[r]+=weight
    for r in range(R):
        out[r]=out[r]/mass[r] if mass[r] else 1/classes
    return out
