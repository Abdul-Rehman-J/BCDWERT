from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import itertools
#cancerdata = pd.read_csv("D:/projectss/data_cancer.csv")
#Load dataset
data = load_breast_cancer()
# Load dataset
dataset = load_breast_cancer()
label_data= data['target_names']
lab = data['target']
feature_selected = data['feature_names']
optimal_features = data['data']
print(label_data)
print(lab[0])
print(feature_selected[0])
print(optimal_features[0])
# Split our data
train_set, test_set, training, testing= train_test_split(optimal_features,lab,test_size=0.25,random_state=41)
# Initialize the classifier
SVC_model= SVC(kernel='linear')
# Train our classifier
model = SVC_model.fit(train_set,training)
SVC_preds = SVC_model.predict(test_set)
print(SVC_preds)
# Evaluate accuracy
accu=(accuracy_score(testing, SVC_preds))
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print("Confusion matrix of the model is :")
print(confusion_matrix(testing,SVC_preds))
print(classification_report(testing,SVC_preds))
True_positive=61
True_Negative=5
False_Positive=6
False_negative=116
True_Positive_Rate=61/(61+116)
False_Positive_Rate=6/(6+116)
Specificity=True_Positive_Rate
Senstivity=1-accu
#print("True positive is: ", True_positive,"\nTrue negative is: ",True_Negative,"\nFalse positive is: ",False_Positive, "\nFalse negative is: ", False_negative)
#print("specificity is : ",Specificity,"\n Senstivity is : ",Senstivity)
# calculate roc curve
print("The Accuracy of the model SVM is :")
print(accu*100,"%")
con=(confusion_matrix(testing,SVC_preds))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("svm")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, SVC_preds)
pyplot.plot(fpr, tpr, marker='.', label='SVM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# Initialize our classifier
GB= GaussianNB()
# Train our classifier
GB_model = GB.fit(train_set, training)
GB_pred= GB.predict(test_set)
# Evaluate accuracy
print("Gaussian Naive bayes accuracy: ", accuracy_score(testing, GB_pred))
print("Confusion matrix of the Gaussian Naive Bayes model is :")
con=(confusion_matrix(testing,GB_pred))
print(con)
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("gb")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)
fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
print(classification_report(testing,GB_pred))
accuracy_all = []
fpr, tpr, thresholds = roc_curve(testing, GB_pred)
pyplot.plot(fpr, tpr, marker='.', label='GNB')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()




# SDG Classifier
SDG = SGDClassifier()
SDG_model=SDG.fit(train_set, training)
SDG_pred = SDG.predict(test_set)
accuracy_all.append(accuracy_score(SDG_pred, testing))
print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(SDG_pred,testing)))
from sklearn.metrics import confusion_matrix
print(classification_report(testing,SDG_pred))
con=(confusion_matrix(testing,SDG_pred))
print(con)
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("sdg")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)
fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('Actual Class Label')
pyplot.xlabel('Predicted Class label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, SDG_pred)
pyplot.plot(fpr, tpr, marker='.', label='SGD')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()







# K-nearest Neighbor Classification

KNN= KNeighborsClassifier()
KNN_model=KNN.fit(train_set, training)
KNN_pred = KNN.predict(test_set)
accuracy_all.append(accuracy_score(KNN_pred, testing))
print("Accuracy of KNN : {0:.2%}".format(accuracy_score(KNN_pred, testing)))
from sklearn.metrics import confusion_matrix
con = confusion_matrix(testing,KNN_pred )
print(classification_report(testing,KNN_pred))
con=(confusion_matrix(testing,KNN_pred))
print(con)
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("knn")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, KNN_pred)
pyplot.plot(fpr, tpr, marker='.', label='KNN')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()







#Random Forest Classifier

RF = RandomForestClassifier()
RF_model=RF.fit(train_set, training)
RF_pred= RF.predict(test_set)
accuracy_all.append(accuracy_score(RF_pred,testing))
print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(RF_pred, testing)))
print(classification_report(testing,RF_pred))
con=(confusion_matrix(testing,RF_pred))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("rf")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, SVC_preds)
pyplot.plot(fpr, tpr, marker='.', label='RF')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



# Extra Trees Classifier
ETC = ExtraTreesClassifier()
ETC_model=ETC.fit(train_set, training)
ETC_pred = ETC.predict(test_set)
accuracy_all.append(accuracy_score(ETC_pred, testing))
print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(ETC_pred, testing)))
print(classification_report(testing,ETC_pred))
con=(confusion_matrix(testing,ETC_pred))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("ext")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, ETC_pred)
pyplot.plot(fpr, tpr, marker='.', label='EXT')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()








#Decision Tree Classifier
DT = DecisionTreeClassifier()
DT_model=DT.fit(train_set, training)
DT_pred = DT.predict(test_set)
accuracy_all.append(accuracy_score(DT_pred, testing))
print("Decision Tree Accuracy: {0:.2%}".format(accuracy_score(DT_pred, testing)))
from sklearn.metrics import confusion_matrix
print(classification_report(testing,DT_pred))
con=(confusion_matrix(testing,DT_pred))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("dt")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, DT_pred)
pyplot.plot(fpr, tpr, marker='.', label='DT')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()








# Logistic Regression

LR = LogisticRegression()
LR_model=LR.fit(train_set, training)
LR_pred = LR.predict(test_set)
accuracy_all.append(accuracy_score(LR_pred, testing))
print("Logistic regression accuracy: {0:.2%}".format(accuracy_score(LR_pred, testing)))
from sklearn.metrics import confusion_matrix
print(classification_report(testing,LR_pred))
con=(confusion_matrix(testing,LR_pred))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("lr")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, LR_pred)
pyplot.plot(fpr, tpr, marker='.', label='LR')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()




# Kernal SVM

KSVM = SVC(kernel = 'rbf', random_state = 0)
KSVM_model=KSVM.fit(train_set, training)
KSVM_pred = KSVM.predict(test_set)
accuracy_all.append(accuracy_score(KSVM_pred, testing))
print("Kernal SVM: {0:.2%}".format(accuracy_score(KSVM_pred, testing)))
from sklearn.metrics import confusion_matrix
print(classification_report(testing,KSVM_pred))
con=(confusion_matrix(testing,KSVM_pred))
classes = [0, 1]
# plot confusion matrix
pyplot.imshow(con, interpolation='nearest', cmap=pyplot.cm.Blues)
pyplot.title("ksvm")
pyplot.colorbar()
tick_marks = np.arange(len(classes))
pyplot.xticks(tick_marks, classes)
pyplot.yticks(tick_marks, classes)

fmt = 'd'
thresh = con.max() / 2.
for i, j in itertools.product(range(con.shape[0]), range(con.shape[1])):
    pyplot.text(j, i, format(con[i, j], fmt),
             horizontalalignment="center",
             color="white" if con[i, j] > thresh else "black")
pyplot.tight_layout()
pyplot.ylabel('True label')
pyplot.xlabel('Predicted label')
pyplot.legend()
pyplot.show()
fpr, tpr, thresholds = roc_curve(testing, KSVM_pred)
pyplot.plot(fpr, tpr, marker='.', label='KSVM')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

