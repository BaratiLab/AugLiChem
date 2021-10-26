import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

X = np.load("./Embed_arr_Augmented_lanths.npy",allow_pickle = True)
labels =  np.load("./Target_arr_Augmented_lanths.npy",allow_pickle = True)

#print(cif_X)

root_dir =  './data/Augmented_lanths/'

# train_file_path = root_dir  + 'id_prop_augment_1.csv'
# df = pd.read_csv(train_file_path,header = None)

#labels =  df[1]
print(labels.shape)

reg = LinearRegression().fit(X,labels)

print(reg.score(X,labels))


prediction_arr = np.load("Embed_arr_abx3_cifs.npy",allow_pickle = True)
pred = reg.predict(prediction_arr)


test_labels = np.load("Target_arr_abx3_cifs.npy",allow_pickle = True)

print(mean_absolute_error(pred,test_labels))



