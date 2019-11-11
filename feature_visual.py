import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv('feature_record_egnn.csv',header=None)
Input_support = []
Input_query = []
Last_layer_supprot = []
Last_layer_query = []
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
        if visual_name == 'Input_feature':
            Input_support.append(X_np[:-5,:])
            Input_query.append(X_np[-5:,:])
        else:
            Last_layer_supprot.append(X_np[:-5,:])
            Last_layer_query.append(X_np[-5:,:])
X_input_support = np.concatenate(Input_support,axis=0)
X_input_query = np.concatenate(Input_query,axis=0)
X_input = np.concatenate([X_input_support,X_input_query],axis=0)
X_lastlayer_support = np.concatenate(Last_layer_supprot,axis=0)
X_lastlayer_query = np.concatenate(Last_layer_query,axis=0)
X_lastlayer = np.concatenate([X_lastlayer_support,X_lastlayer_query],axis=0)
label_support = np.tile(label[:-5],10)
label_query = np.tile(label[-5:],10)
label = np.concatenate([label_support,label_query],axis=0)

X_input_emb = TSNE(n_components=2,perplexity=59).fit_transform(X_input)
X_lastlayer_emb = TSNE(n_components=2,perplexity=59).fit_transform(X_lastlayer)

# input
plt.figure(1)
p1 = plt.scatter(X_input_emb[:-50,0], X_input_emb[:-50,1], c=label[:-50])
p2 = plt.scatter(X_input_emb[-50:,0],X_input_emb[-50:,1],c=label[-50:],marker='*')
plt.legend((p1, p2), ('support','query'))
plt.title('Input_feature')
plt.figure(2)
p1 = plt.scatter(X_lastlayer_emb[:-50,0], X_lastlayer_emb[:-50,1], c=label[:-50])
p2 = plt.scatter(X_lastlayer_emb[-50:,0],X_lastlayer_emb[-50:,1],c=label[-50:],marker='*')
plt.legend((p1, p2), ('support','query'))
plt.title('Last_layer')
plt.show()

