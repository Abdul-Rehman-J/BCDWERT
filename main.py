
from sklearn.datasets import load_breast_cancer
#cancerdata = pd.read_csv("D:/projectss/data_cancer.csv")
#Load dataset
data = load_breast_cancer()
from sklearn.datasets import load_breast_cancer
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
from sklearn.model_selection import train_test_split
# Split our data
train_set, test_set, training, testing= train_test_split(optimal_features,lab,test_size=0.25,random_state=50)
from sklearn.svm import SVC
# Initialize the classifier
prediction_model= SVC(kernel='linear')
# Train our classifier
model = prediction_model.fit(train_set, training)
predictions = prediction_model.predict(test_set)
print(predictions)
from sklearn.metrics import accuracy_score
# Evaluate accuracy
accu=(accuracy_score(testing, predictions))
from sklearn.metrics import classification_report, confusion_matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print("Confusion matrix of the model is :")
print(confusion_matrix(testing,predictions))
print(classification_report(testing,predictions))
print("The Accuracy of the model is :")
print(accu*100,"%")
True_positive=61
True_Negative=5
False_Positive=6
False_negative=116
True_Positive_Rate=61/(61+116)
False_Positive_Rate=6/(6+116)
Specifity=True_Positive_Rate
Senstivity=1-accu
Precision = True_positive / (True_positive + False_Positive)
Recall = True_positive / (True_positive + False_negative)
F_measure = (2 * Precision * Recall) / (Precision + Recall)
print("True positive is: ", True_positive,"\nTrue negative is: ",True_Negative,"\nFalse positive is: ",False_Positive, "\nFalse negative is: ", False_negative)
print("specificity is : ",Specifity,"\n Senstivity is : ",Senstivity,"\nPrecision is : " ,Precision," Recall is : ",Recall, "\nF-Measure is : ", F_measure)
# calculate roc curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
fpr, tpr, thresholds = roc_curve(testing, predictions)
pyplot.plot(fpr,tpr, linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
