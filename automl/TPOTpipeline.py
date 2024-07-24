import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from tpot.export_utils import set_param_recursive

data = np.load('../data/test_curated.npz',allow_pickle=True)
x_test = data['test']
LABELS_SUB = data['labels']

data = np.load('../data/train_curated.npz',allow_pickle=True)
x_train = data['train']
y_train = data['target']

print('data prepared')

exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    SelectFwe(score_func=f_classif, alpha=0.048),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=18, min_samples_split=20, n_estimators=100)
)
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

print('pipeline ready, fitting...')
exported_pipeline.fit(x_train, y_train)

print('fitted, predicting...')
results = exported_pipeline.predict(x_test)

print('done')

predictions = pd.DataFrame(results,columns=['Transported'])
predictions['PassengerId'] = LABELS_SUB
predictions['Transported'] = predictions['Transported'].astype(bool)
predictions = predictions[['PassengerId','Transported']]
predictions.to_csv('../data/submissionTPOT.csv',index=False)
