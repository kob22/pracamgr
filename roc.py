from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from data import importdata
import numpy as np
from sklearn import naive_bayes
from sklearn import tree

dataset = ['load_german', 'load_haberman', 'load_transfusion', 'load_ionosphere', 'load_balance_scale', 'load_bupa',
           'load_car', 'load_cmc', 'load_ecoli',
           'load_glass', 'load_new_thyroid', 'load_seeds', 'load_solar_flare', 'load_vehicle', 'load_vertebal',
           'load_yeastME1', 'load_yeastME2', 'load_yeastME3',
           'load_abalone0_4', 'load_abalone16_29', 'load_abalone0_4_16_29']
db = getattr(importdata, dataset[20])()
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', naive_bayes.GaussianNB())])
pipe_lr2 = Pipeline([('scl', StandardScaler()),
                     ('clf', LogisticRegression(penalty='l2',
                                                random_state=0,
                                                C=1.0))])

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, test_size=0.3, stratify=db.target,
                                                    random_state=5)

fig = plt.figure(figsize=(7, 5), facecolor='white')

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

probas = pipe_lr.fit(X_train,
                     y_train).predict_proba(X_test)

probas2 = pipe_lr2.fit(X_train,
                       y_train).predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test,
                                 probas[:, 1],
                                 pos_label=1)
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr,
         tpr,
         lw=2,
         label='ROC NB, (AUC = %0.2f)'
               % (roc_auc))

fpr, tpr, thresholds = roc_curve(y_test,
                                 probas2[:, 1],
                                 pos_label=1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr,
         tpr,
         lw=2,
         label='ROC LR, (AUC = %0.2f)'
               % (roc_auc))
plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='losowe zgadywanie')

mean_tpr /= 1
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot([0, 0, 1],
         [0, 1, 1],
         lw=2,
         linestyle=':',
         color='black',
         label='idealna klasyfikacja')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')

plt.tight_layout()
# plt.savefig('./figures/roc.png', dpi=300)
plt.show()
plt.savefig('roc.png')
