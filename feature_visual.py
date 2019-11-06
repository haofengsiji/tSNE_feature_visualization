import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv('feature_record_hgnn.csv',header=None)
for i in range(data.shape[0]):
    X_ls = []
    if data.iloc[i,0] == 'label':
        label = data.iloc[i,1:].values
        if type(label[0]) is str:
            label_ls = []
            for l in label:
                if type(l) is float:
                    break
                else:
                    label_ls.append(np.array(eval(l)).astype(np.uint8))
            label = np.stack(label_ls,axis=0)
    else:
        visual_name = data.iloc[i,0]
        X = data.iloc[i,1:].values
        for x in X:
            if type(x) is float:
                break
            else:
                X_ls.append(np.fromstring(x.replace('[', '').replace(']', '').replace('\n', ' '), sep=' '))
        X_np = np.stack(X_ls,axis=0)
        prep_value = int(label.shape[0]/5)-1 # trand -1 / non -0
        X_embedded = TSNE(n_components=2,perplexity=5).fit_transform(X_np)
        plt.figure(i)
        p1 = plt.scatter(X_embedded[:-5,0], X_embedded[:-5,1], c=label[:-5])
        p2 = plt.scatter(X_embedded[-5:,0],X_embedded[-5:,1],c=label[-5:],marker='*')
        plt.legend((p1, p2), ('support','query'))
        plt.title(visual_name)
        plt.xlabel('x')
        plt.ylabel('y')
plt.show()

