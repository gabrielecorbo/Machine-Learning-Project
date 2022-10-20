import numpy as np 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn import svm
import seaborn as sn
import time
import pickle
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, plot_roc_curve, confusion_matrix, precision_score, recall_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

data_orig = pd.read_csv('Project_Data_EE4C12_SET_PV.csv') 
y_2c=data_orig.iloc[:,6]
data = data_orig.drop('Fault_Type', axis=1) # remove the ordinary encoding column
data['Healthy'] = np.where(data_orig['Fault_Type'] == 0, 1, 0)
data['Short_Circuit'] = np.where(data_orig['Fault_Type'] == 1, 1, 0)
data['Broken_Cells'] = np.where(data_orig['Fault_Type'] == 2, 1, 0)
data['Broken_Strings'] = np.where(data_orig['Fault_Type'] == 3, 1, 0)
training_set, testing_set = train_test_split(data, test_size=0.15, random_state=4720)

X_training = training_set[['Irradiance', 'Ambient_Temperature', 'Sun_Azimuth', 'Sun_Elevation', 
                              'System_Power', 'System_Age']]
y_2c_training = training_set['System_Status']
y_mc_training = training_set[['Healthy', 'Short_Circuit', 'Broken_Cells', 'Broken_Strings']]

X_test = testing_set[['Irradiance', 'Ambient_Temperature', 'Sun_Azimuth', 'Sun_Elevation', 
                              'System_Power', 'System_Age']]
y_2c_test = testing_set['System_Status']
y_mc_test = testing_set[['Healthy', 'Short_Circuit', 'Broken_Cells', 'Broken_Strings']]

scaler = StandardScaler()
scaler.fit(X_training)
X_training_scaled = scaler.transform(X_training)

#X_train, X_val, y_train_2c, y_val_2c = train_test_split(X_training_scaled, y_2c_training, test_size=0.15, random_state=4720)
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(degree=3)
pol.fit(X_training_scaled)
X_train_pol = pol.transform(X_training_scaled)

layers=3
nodes=[40,20,10]
input_node=84
output_node=2
model = Sequential()
model.add(Dense(nodes[0], input_dim=input_node, activation='relu'))
for i in range(1,layers):
    model.add(Dense(nodes[i], activation='relu'))
model.add(Dense(output_node, activation='softmax'))
# Compile mode
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#encoder = LabelEncoder()
#encoder.fit(y_val_2c)
#encoded_Y = encoder.transform(y_val_2c)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y_val = np_utils.to_categorical(encoded_Y)

encoder2 = LabelEncoder()
encoder2.fit(y_2c_training)
encoded_Y_t = encoder2.transform(y_2c_training)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_t)

history=model.fit(X_train_pol, dummy_y_train, epochs=600, batch_size=10,validation_split=0.15)
#_, accuracy = model.evaluate(X_val_pol, dummy_y_val)
#print('Accuracy: %.2f' % (accuracy*100))

history_dict = history.history

# learning curve
# accuracy
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
print(val_acc.pop())
# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# range of X (no. of epochs)
epochs = range(1, len(acc) + 1)

# plot
# "r" is for "solid red line"
plt.plot(epochs, acc, 'r', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()