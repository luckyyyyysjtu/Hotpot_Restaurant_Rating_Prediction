import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from scipy import stats


filename = './datanew.csv'
dataframe = pd.read_csv(filename)
dataframe = dataframe.fillna(dataframe.mean())
dataset = dataframe.to_numpy()
X = dataset[:, 2:13]
y = dataset[:, -1]
y = y.astype('int')
X = X.astype('float')


C = [0.001, 0.01, 0.1, 1, 5, 10]
ave_test_accuracy = []
sd_test_accuracy = []
ave_margin = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)
# kf = KFold(n_splits=10, shuffle=False)
#
# for c in C:
#     print('1')
#     margin = []
#     test_accuracy = []
#     for train_index, test_index in kf.split(X_train):
#         X_train2, y_train2 = X_train[train_index], y_train[train_index]
#         X_test2, y_test2 = X_train[test_index], y_train[test_index]
#         clf = svm.SVC(C=c, kernel='linear')
#         clf.fit(X_train2, y_train2)
#
#         test_accuracy_element = clf.score(X_test2, y_test2)
#         test_accuracy.append(test_accuracy_element)
#         margin_element = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
#         margin.append(margin_element)
#
#     ave_test_accuracy_element = np.mean(test_accuracy)
#     ave_test_accuracy.append(ave_test_accuracy_element)
#     sd_test_accuracy_element = np.std(test_accuracy, ddof=1)
#     sd_test_accuracy.append(sd_test_accuracy_element)
#     ave_margin_element = np.mean(margin)
#     ave_margin.append(ave_margin_element)
#
# plt.plot(C, ave_test_accuracy, label="avg. accuracy")
# plt.xlabel("The penalty coefficient")
# plt.ylabel("Avg. accuracy")
# plt.title("The penalty coefficient vs the avg. accuracy")
# plt.legend()
# plt.show()
#
#
# plt.plot(C, sd_test_accuracy, label="std. dev. of accuracy")
# plt.xlabel("The penalty coefficient")
# plt.ylabel("std. dev. of accuracy")
# plt.title("The penalty coefficient vs the std. dev. of accuracy")
# plt.legend()
# plt.show()
#
#
# plt.plot(C, ave_margin, label="avg. margin")
# plt.xlabel("The penalty coefficient")
# plt.ylabel("Avg. margin")
# plt.title("The penalty coefficient vs the avg. margin")
# plt.legend()
# plt.show()


# choose c=1
clf = svm.SVC(C=1, kernel='linear', probability=True)
clf.fit(X_train, y_train)
# result = clf.predict(X_test)
# print('result: ', result)
print("The accuracy on test data is %s" % clf.score(X_test, y_test))
print('b: ', clf.intercept_)
print('w: ', clf.coef_)
probs_y = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probs_y[:, 1])
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# significance
params = np.append(clf.intercept_, clf.coef_)
predictions = clf.predict(X_test)
newX_test = pd.DataFrame({"Constant": np.ones(len(X_test))}).join(pd.DataFrame(X_test))
MSE = (sum(y_test-predictions)**2)/(len(newX_test)-len(newX_test.columns))
var_b = MSE * (np.linalg.inv(np.dot(newX_test.T, newX_test)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params / sd_b

p_values = [2*(1-stats.t.cdf(np.abs(i), (len(newX_test)-1))) for i in ts_b]

sd_b = np.round(sd_b, 5)
ts_b = np.round(ts_b, 3)
p_values = np.round(p_values, 5)
params = np.round(params, 4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                              p_values]
print(myDF3)


# confusion matrix
sns.set()
f, ax = plt.subplots()
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
C2 = confusion_matrix(y_test, predictions, labels=[0, 1])
print(C2)
sns.heatmap(C2, annot=True, ax=ax)
plt.show()
