import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics


main_path = './adult_dataset/adult_dataset/'
adult_test_feature = main_path + 'adult_test_feature.txt'
adult_test_label = main_path + 'adult_test_label.txt'
adult_train_feature = main_path + 'adult_train_feature.txt'
adult_train_label = main_path + 'adult_train_label.txt'


class AdaBoost(object):
    def __init__(self, n_estimators=10):
        super().__init__()
        self.classfiers = []    #子分类器
        self.n_estimators = n_estimators

    def stump_classfier(self, train_feature, train_label, D):
        clf = DecisionTreeClassifier()
        clf = clf.fit(train_feature, train_label, sample_weight=D)
        predict_label = clf.predict(train_feature)
        w_error = np.ones(len(predict_label))
        w_error[predict_label == train_label] = 0
        w_error = D.T * w_error
        sum_error = np.sum(w_error)
        return clf, sum_error, predict_label

    def fit(self, train_feature, train_label):
        self.classfiers = []
        m = len(train_label)
        D = np.ones(m) / m
        train_label_copy = train_label.copy()
        train_label_copy[train_label_copy == 0] = -1
        train_label = train_label_copy
        for t in range(self.n_estimators):
            clf, w_error, predict_label = self.stump_classfier(train_feature, train_label, D)
            alpha = 0.5 * np.log((1 - w_error) / max(w_error, 1e-16))  # 计算分类器权重
            # print('t=%d, error=%f, alpha=%f' % (t, w_error, alpha))
            if w_error > 0.5:
                break
            self.classfiers.append((clf, alpha))  # 加入分类器
            D = D * np.exp(-alpha * predict_label * train_label)
            D = D / np.sum(D)
        return self

    def predict(self, test_feature):
        h = np.zeros(len(test_feature))
        for i in range(len(self.classfiers)):
            clf = self.classfiers[i]
            predict_label = clf[0].predict(test_feature)
            predict_label = predict_label * clf[1]
            h = h + predict_label
        H = np.sign(h)
        H[H == -1] = 0
        return H

    def predict_proba(self, test_feature):
        proba = np.zeros(len(test_feature))
        sum_alpha = .0
        for pair in self.classfiers:
            clf = pair[0]
            alpha = pair[1]
            sum_alpha += alpha
            predict_proba = clf.predict_proba(test_feature)
            idx_posClass = np.argwhere(clf.classes_== 1)
            pos_proba = np.zeros(len(test_feature))
            for i in range(len(predict_proba)):
                pos_proba[i] = predict_proba[i][idx_posClass] * alpha
            # print(predict_proba)
            proba += pos_proba
        return proba / sum_alpha


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


def rate(test_label, predict_label):
    nr_pos = 0
    nr_neg = 0
    for (item1, item2) in zip(test_label, predict_label):
        if item1 == item2:
            nr_pos += 1
        else:
            nr_neg += 1
    nr_total = nr_pos + nr_neg
    acc = float(nr_pos)/nr_total
    print('accuracy=%d/%d=%f'%(nr_pos, nr_total, acc))
    return acc


def test_adaBoost(ada_clf, test_feature, test_label):
    h = np.zeros(len(test_feature))
    N = len(ada_clf.classfiers)
    acc_list = []
    for i in range(N):
        clf = ada_clf.classfiers[i]
        predict_label = clf[0].predict(test_feature)
        predict_label = predict_label * clf[1]
        h = h + predict_label
        H = np.sign(h)
        H[H == -1] = 0
        acc = rate(H, test_label)
        acc_list.append(acc)
    x = np.linspace(1, N, N)
    plt.plot(x, acc_list, 'r-o', label='acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy-Epoch')
    plt.savefig(main_path + "accuracy_" + str(N) + ".png")
    plt.show()



if __name__ == '__main__':
    train_feature, train_label = load_adult(adult_train_feature, adult_train_label)
    test_feature, test_label = load_adult(adult_test_feature, adult_test_label)
    ada_boost = AdaBoost(n_estimators=28)
    ada_boost.fit(train_feature, train_label)
    predict_proba = ada_boost.predict_proba(test_feature)
    auc = metrics.roc_auc_score(test_label, predict_proba)
    print('AUC =',auc)
