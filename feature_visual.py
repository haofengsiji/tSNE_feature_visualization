import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.manifold import TSNE

data = pd.read_csv('feature_record.csv',header=None)
c = []
for i in range(data.shape[0]):
    if data.iloc[i,0] == 'label':
        c = data.iloc[i,0]
    else:
        visual_name = data.iloc[i,0]
        X = data.iloc[i,1:].values
        X_np = [np.fromstring(x.replace('[', '').replace(']', '').replace('\n', ' '), sep=' ') for x in X ]
        X_np = np.stack(X_np,axis=0)
        X_embedded = TSNE(n_components=2).fit_transform(X_np)
