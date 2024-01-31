import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import tqdm
import copy
import itertools

#读取数据
filename = 'datanew.csv'
dataframe = pd.read_csv(filename)
dataframe = dataframe.fillna(dataframe.mean())
dataset = dataframe.to_numpy()
X = dataset[:, 2:13]
y = dataset[:,-1]

#总共11个特征，输出组合情况
feature_list = []
for i in range(1,12):
    feature_list += list(itertools.combinations([2,3,4,5,6,7,8,9,10,11,12],i)) 

ave_train = []
std_train = []
ave_test = []
std_test = []

#循环计算每一种组合情况下的准确度均值、标准差
for fe in feature_list:
    print('当前特征为:',fe)
    accuracy_test = []
    accuracy_train = []
    X = dataset[:, fe]
    for random in tqdm(range(1,5)):
        X_train, X_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size = 0.3, random_state = random)

        LR = lr(solver = 'liblinear') 
        LR.fit(X_train, y_train) 
        
        test_accuracy = LR.score(X_test, y_test)
        train_accuracy = LR.score(X_train, y_train)
        accuracy_test.append(test_accuracy)
        accuracy_train.append(train_accuracy)

    average_accuracy_train = np.mean(accuracy_train)
    average_accuracy_test = np.mean(accuracy_test)
    ave_train.append(average_accuracy_train)
    ave_test.append(average_accuracy_test)

    print('The average accuracy on training data:')
    print(np.around(average_accuracy_train,decimals=3))
    print('The average accuracy on test data:')
    print(np.around(average_accuracy_test,decimals=3))

    std_accuracy_train = np.std(accuracy_train)
    std_accuracy_test = np.std(accuracy_test)
    std_train.append(std_accuracy_train)
    std_test.append(std_accuracy_test)

#选取所有2047个组合中训练集平均准确度最高的10个组合
#找出它们在训练集、测试集上的准确度均值、标准差
#输出它们的序号、组合中的特征、在训练集、测试集上的准确度均值
t = copy.deepcopy(ave_train)
max_number_train = []
max_index_train = []
max_feature_train = []
max_number_test = []
std_max_number_train = []
std_max_number_test = []
for _ in range(10):
    number = max(t)
    index = t.index(number)
    t[index] = 0
    max_number_train.append(number)
    max_index_train.append(index)
    max_feature_train.append(feature_list[index])
    max_number_test.append(ave_test[index])
    std_max_number_train.append(std_train[index])
    std_max_number_test.append(std_test[index])
t = []
print('The Index of Large Accuracy on Training Data: ')
print(max_index_train)
print('The Features of Large Accuracy on Training Data: ')
print(max_feature_train)
print('The Value of Large Accuracy on Training Data: ')
print(np.around(max_number_train,decimals=3))
print('The Value of Large Accuracy on Test Data: ')
print(np.around(max_number_test,decimals=3))

#将前10名组合在训练集、测试集上的准确度均值可视化
plt.figure(figsize = (50,8),dpi = 80) 
bar_width = 0.4
tick_label = max_feature_train
i = np.arange(len(max_number_train))
plt.bar(i,max_number_train,bar_width,align = "center",color = "c",label = "Accuracy on training data",alpha = 0.5)
plt.bar(i+bar_width,max_number_test,bar_width,color = "b",align = "center",label = "Accuracy on test data",alpha = 0.5)
plt.xlabel("Features")
plt.ylabel("Accuracy")
plt.title("Accuracy with differenct features")
plt.xticks(i+bar_width/2,tick_label)
plt.legend()
plt.show()

#将前10名组合在训练集、测试集上的准确度标准差可视化
plt.figure(figsize = (50,8),dpi = 80) 
bar_width = 0.4
tick_label = max_feature_train
i = np.arange(len(max_number_train))
plt.bar(i,std_max_number_train,bar_width,align = "center",color = "c",label = "Std of Accuracy on training data",alpha = 0.5)
plt.bar(i+bar_width,std_max_number_test,bar_width,color = "b",align = "center",label = "Std of Accuracy on test data",alpha = 0.5)
plt.xlabel("Features")
plt.ylabel("Std of Accuracy")
plt.title("Std of Accuracy with differenct features")
plt.xticks(i+bar_width/2,tick_label)
plt.legend()
plt.show()


#以前3名的组合为例，作ROC曲线
X = np.hstack((dataset[:,3:4],dataset[:,7:9]))
X_train, X_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size = 0.3, random_state = 2)
LR = lr(solver = 'liblinear')
LR.fit(X_train, y_train)
probs_y = LR.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test,probs_y[:, 1])
plt.plot(fpr,tpr,"b",label = "ROC")
plt.plot([0,1],[0,1],"k",linestyle = '--',label = "random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve 1")
plt.legend()
plt.show()

X = np.hstack((dataset[:,3:4],dataset[:,8:9]))
X_train, X_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size = 0.3, random_state = 3)
LR = lr(solver = 'liblinear')
LR.fit(X_train, y_train)
probs_y = LR.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test,probs_y[:, 1])
plt.plot(fpr,tpr,"b",label = "ROC")
plt.plot([0,1],[0,1],"k",linestyle = '--',label = "random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve 2")
plt.legend()
plt.show()

X = np.hstack((dataset[:,4:6],dataset[:,7:9]))
X_train, X_test, y_train, y_test = train_test_split(X, y.astype('int'), test_size = 0.3, random_state = 4)
LR = lr(solver= 'liblinear')
LR.fit(X_train, y_train)
probs_y = LR.predict_proba(X_test)
fpr,tpr,thresholds = roc_curve(y_test,probs_y[:, 1])
plt.plot(fpr,tpr,"b",label = "ROC")
plt.plot([0,1],[0,1],"k",linestyle = '--',label = "random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve 3")
plt.legend()
plt.show()