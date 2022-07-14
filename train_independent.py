from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy
from tqdm import tqdm
import pickle
import os
from pandas import read_csv
from gym_malware.envs.utils.pefeatures2 import PEFeatureExtractor2

path = '/home/containernet/KLTN/Pesidious/CSV'
print ("Load benign train dataset.")
benign_train_data = read_csv(os.path.join(path, 'benign_train_data.csv'), header = None)
print ("Load malware train dataset.")
malware_train_data = read_csv(os.path.join(path, 'malware_train_data.csv'), header = None)

print ("Load benign test dataset")
benign_test_data = read_csv(os.path.join(path, 'benign_test_data.csv'), header = None)
print ("Load malware test data")
malware_test_data = read_csv(os.path.join(path, 'malware_test_data.csv'), header = None)

benign_train_data = benign_train_data.to_numpy()
malware_train_data = malware_train_data.to_numpy()
benign_test_data = benign_test_data.to_numpy()
malware_test_data = malware_test_data.to_numpy()

print ("Concatenate benign_malware.")
x = numpy.concatenate([benign_train_data, malware_train_data])
x_test = numpy.concatenate([benign_test_data, malware_test_data])

del benign_train_data, malware_train_data, benign_test_data, malware_test_data
y = numpy.concatenate([numpy.zeros(40000), numpy.ones(40000)])
y_test = numpy.concatenate([numpy.zeros(10000), numpy.ones(10000)])

# print ("Split train_valid")
# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2)

# define some single learner
dt = DecisionTreeClassifier()
logistic = LogisticRegression(solver='lbfgs', max_iter=10000)
naive = GaussianNB()
mlp = MLPClassifier(max_iter=100)
knn = KNeighborsClassifier()

single_list = [logistic, dt, naive, mlp, knn]

# define some ensemble learner
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
gradient = GradientBoostingClassifier()
bagging = BaggingClassifier()
single_base = [('dt', dt), 
                    ('lr', logistic), 
                    ('nai', naive), 
                    ('mlp', mlp), 
                    ('knn', knn)]

voting = VotingClassifier(estimators=single_base, voting='soft', n_jobs=-1)

bagging_boosting_list = [rf, ada, gradient, bagging, voting]

ensemble_base = [('dt', dt), 
                    ('rf', rf), 
                    ('ada', ada), 
                    ('gradient', gradient), 
                    ('baggin', bagging)]

# define stacking

stacking_single = StackingClassifier(estimators=single_base, final_estimator=DecisionTreeClassifier(), n_jobs=None, cv=10)
stacking_ensemble = StackingClassifier(estimators=ensemble_base, final_estimator=RandomForestClassifier(), n_jobs=None, cv=10)

stacking_list = [stacking_single, stacking_ensemble]

results_file = Path('CSV/results.csv')
exists = results_file.exists()

auc_list = list()

with open(str(results_file), "a+") as f_out:
        header = ",".join(["Name", "AUC", "Accuracy", "Precision", "Recall", "F1-Score",])
        if not exists:
            f_out.write(header)

        print ("Training.")

        for clf in tqdm(stacking_list, desc="Progess: "):
            print (clf.__class__.__name__[0:8] + clf.final_estimator.__class__.__name__)
            result = list()
            clf.fit(x, y)
            y_pred = clf.predict(x_test)

            y_prob = clf.predict_proba(x_test)[::, 1]
            pickle.dump(clf, open('CSV/' + clf.__class__.__name__ + '.pkl', 'wb'))
            auc = "%.5f" % roc_auc_score(y_test, y_prob)
            acc = "%.5f" % accuracy_score(y_test, y_pred)
            precision = "%.5f" % precision_score(y_test, y_pred)
            recall = "%.5f" % recall_score(y_test, y_pred)
            f1 = "%.5f" % f1_score(y_test, y_pred)
            result.extend(('\n' + clf.__class__.__name__[0:8] + clf.final_estimator.__class__.__name__, auc, acc, precision, recall, f1))
            result = ",".join(result)
            f_out.write(result)
            disp = plot_confusion_matrix(clf, x_test, y_test, normalize='true', display_labels=['Benign', 'Malware'], values_format='.3g', cmap="Blues")
            disp.plot()
            disp.figure_.savefig('CSV/confusion_' + clf.__class__.__name__[0:8] + clf.final_estimator.__class__.__name__[:-10] + '.png')
            disp.figure_.clf()
        
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_list.append([fpr, tpr, clf.__class__.__name__[0:8] + clf.final_estimator.__class__.__name__[:-10] + ", AUC="+str(auc)])


for auc in auc_list:
    plt.plot(auc[0], auc[1], label=auc[2])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)
plt.savefig("CSV/auc.png")
