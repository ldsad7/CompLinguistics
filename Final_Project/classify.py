from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score, roc_curve, auc, \
    precision_recall_curve, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.externals import joblib
import csv
import numpy
from scipy import interp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def import_csv(path):
    """
    Import feature data from a given csv file
    :param path: path to CSV file containing tokens and features
    :return features and target tags as numpy arrays
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # skip header
        data = []
        target = []
        for row in reader:
            data.append(row[:-1])
            target.append(row[-1])
        # convert to numpy arrays
        data = numpy.array(data)
        target = numpy.array(target)
    return data, target
    

def import_as_dict(path):
    """
    Import feature data from a given csv file
    :param path: path to CSV file containing tokens and features
    :return features and target tags as sparse matrices
    Should deal with categorical input.
    """
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        data = []
        target = []

        # read things from csv
        for row in reader:
            data.append(dict(zip(header[:-1], row[:-1])))  # make a dict of feature : value
            target.append(row[-1])
        from sklearn.feature_extraction import DictVectorizer
        vec = DictVectorizer()

        # convert categorical features to floats
        data_matrix = vec.fit_transform(data)
        v = vec.transform(data[1])

        # convert targets to numpy array as strings
        target_matrix = numpy.array(target)

        # save converter to use in prediction
        #joblib.dump(vec, 'feature_transformer.pkl')
    #target_names = set(target)
    joblib.dump(vec, 'feature_transformer.pkl')
    return data_matrix, target_matrix #, target_names
    

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # import data
    print("Import data")
    data, target = import_as_dict('roles_syntax_predicates_and_others.csv')
    print(data.shape, target.shape)

    # split data into train and test subsets
    print("Split data")
    data_train, data_test, target_train, target_test = train_test_split(data, target,  random_state=42)

    classifiers = [MultinomialNB(),
                   SGDClassifier(penalty='elasticnet', eta0=0.00390625, learning_rate='constant', alpha=1e-06, loss='hinge', random_state=42),
                   KNeighborsClassifier(5),
                   DecisionTreeClassifier(random_state=42),
                   RandomForestClassifier(n_estimators=100, random_state=42)]
                   MLPClassifier()]
                   
    names = ["MNB","SGD","KNN","DT","RF", "MLP"]

    
    labels_ = classifiers[0].classes_[2:]
    for name, clf in zip (names, classifiers):
        y_score = clf.fit(data_train, target_train)
        y_true, y_pred = target_test, clf.predict(data_test)
        labels_ = clf.classes_[2:]
        #print(classification_report(y_true, y_pred))
        precision, recall, fscore, support = score(y_true, y_pred)
        #print(precision, recall, fscore, clf.classes_)
        plt.plot(labels_, precision, label='{}'.format(name))
    plt.xticks(rotation=90, fontsize=5)
    plt.title("Precision")
    plt.legend(loc='upper right')
    plt.savefig('Precision.png', figsize=(30,30), dpi=200, bbox_inches='tight')
    plt.show()
    print("Precision")

    for name, clf in zip (names, classifiers):
        y_score = clf.fit(data_train, target_train)
        y_true, y_pred = target_test, clf.predict(data_test)
        precision, recall, fscore, support = score(y_true, y_pred)
        plt.plot(labels_, recall, label='{}'.format(name))
    plt.xticks(rotation=90, fontsize=5)
    plt.title("Recall")
    plt.legend(loc='upper left')
    plt.savefig('Recall.png', figsize=(30,30), dpi=200, bbox_inches='tight')
    plt.show()
    print("Recall")

    for name, clf in zip (names, classifiers):
        y_score = clf.fit(data_train, target_train)
        y_true, y_pred = target_test, clf.predict(data_test)
        precision, recall, fscore, support = score(y_true, y_pred)
        plt.plot(labels_, fscore, label='{}'.format(name))
    plt.xticks(rotation=90, fontsize=5)
    plt.title("Fscore")
    plt.legend(loc='upper left')
    plt.savefig('Fscore.png', figsize=(30,30), dpi=200, bbox_inches='tight')
    plt.show()
    print("Fscore")
    
    joblib.dump(clf, 'frame_parser.pkl')
