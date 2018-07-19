import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics


main_path = './adult_dataset/adult_dataset/'
adult_test_feature = main_path + 'adult_test_feature.txt'
adult_test_label = main_path + 'adult_test_label.txt'
adult_train_feature = main_path + 'adult_train_feature.txt'
adult_train_label = main_path + 'adult_train_label.txt'

def load_adult(train_feature_path, train_label_path):
    train_feature = []
    train_label = []
    with open(train_feature_path) as fp:
        lines = fp.readlines()
        for line in lines:
            feature = list(map(float, line.split(' ')))
            train_feature.append(feature)
    with open(train_label_path) as fp:
        lines = fp.readlines()
        for line in lines:
            label = int(line)
            train_label.append(label)
    train_feature = np.array(train_feature)
    train_label = np.array(train_label)
    return train_feature, train_label


class RandomForest(object):
    def __init__(self, n_estimators=10, max_features='log2', ratio=1.0):
        super().__init__()
        self.classfiers = []
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.ratio = ratio

    def stump_classfier(self, train_feature, train_label):
        clf = DecisionTreeClassifier()
        clf = clf.fit(train_feature, train_label)
        # predict_label = clf.predict(train_feature)
        return clf

    def bootstrap(self, train_feature, train_label):
        samples_feature = []
        samples_label = []
        if isinstance(self.max_features, str):
            k = int(np.log2(len(train_feature[0])-1))
        elif isinstance(self.max_features, float):
            k = int((len(train_feature[0])-1) * self.max_features)
        elif isinstance(self.max_features, int):
            k = self.max_features
        else:
            raise RuntimeError('incorrect arg of max_features')
        n_feature = len(train_feature)
        n_samples = round(n_feature * self.ratio)
        k_random = random.sample(range(len(train_feature[0])-1), k)
        while len(samples_label) < n_samples:
            idx = np.random.randint(n_feature)
            sample = train_feature[idx]
            samples_feature.append([sample[i] for i in k_random])
            samples_label.append(train_label[idx])
        return samples_feature, samples_label, k_random

    def fit(self, train_feature, train_label):
        self.classfiers = []
        for i in range(self.n_estimators):
            samples_feature, samples_label, k_random = self.bootstrap(train_feature, train_label)
            clf = self.stump_classfier(samples_feature, samples_label)
            self.classfiers.append((clf, k_random))

    def predict(self, test_feature):
        predict_label = []
        votes = np.zeros(len(test_feature))
        for pair in self.classfiers:
            clf = pair[0]
            k_random = pair[1]
            features = []
            for feature in test_feature:
                features.append([feature[i] for i in k_random])
            label = clf.predict(features)
            votes += label
        for i in range(len(test_feature)):
            if votes[i] > len(self.classfiers)/2:
                predict_label.append(1)
            elif votes[i] == len(self.classfiers)/2:
                r = np.random.random()
                if r < 0.5:
                    predict_label.append(1)
                else:
                    predict_label.append(0)
            else:
                predict_label.append(0)
        return predict_label

    def predict_proba(self, test_feature):
        '''
        :param test_feature:
        :return: proba of pos class
        '''
        proba = np.zeros(len(test_feature))
        for pair in self.classfiers:
            clf = pair[0]
            k_random = pair[1]
            features = []
            for feature in test_feature:
                features.append([feature[i] for i in k_random])
            predict_proba = clf.predict_proba(features)
            idx_posClass = np.argwhere(clf.classes_ == 1)
            pos_proba = np.zeros(len(test_feature))
            for i in range(len(predict_proba)):
                pos_proba[i] = predict_proba[i][idx_posClass]
            # print(predict_proba)
            proba += pos_proba
        return proba / len(self.classfiers)


def rate(predict_label, test_label):
    if not isinstance(predict_label, np.ndarray):
        predict_label = np.array(predict_label)
    if not isinstance(test_label, np.ndarray):
        test_label = np.array(test_label)
    test_array = np.zeros(len(predict_label))
    test_array[predict_label == test_label] = 1
    nr_pos = np.sum(test_array)
    return nr_pos/len(predict_label)



if __name__ == '__main__':
    train_feature, train_label = load_adult(adult_train_feature, adult_train_label)
    test_feature, test_label = load_adult(adult_test_feature, adult_test_label)
    randomForest_clf = RandomForest(n_estimators=30, max_features=0.6, ratio=0.7)
    randomForest_clf.fit(train_feature, train_label)
    predict_proba = randomForest_clf.predict_proba(test_feature)
    auc = metrics.roc_auc_score(test_label, predict_proba)
    print('AUC =', auc)