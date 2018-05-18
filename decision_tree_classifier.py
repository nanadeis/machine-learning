"""
    用于分类的决策树
    date：2018/5/15
"""
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

def get_feature_type(X):
    feature_type = []
    for i in range(X.shape[1]):
        feature_value = [x[i] for x in X]
        values = np.unique(feature_value)
        if len(values) > 5:
            feature_type.append('c')
        else:
            feature_type.append('d')
    return feature_type


def gini(labels):
    _, a = np.unique(labels, return_counts=True)
    return 1 - np.sum((a / np.sum(a)) ** 2)


def continuous_gini(x, y):
    x_u = np.unique(x)
    sep_vals = (x_u[:-1] + x_u[1:]) / 2  # 技巧！
    min_gini =1
    best_sep_val = None
    for sep_val in sep_vals:
        temp_gini = (y[x <= sep_val].shape[0] * gini(y[x<= sep_val]) + y[x > sep_val].shape[0] * gini(y[x > sep_val])) / x.shape[0]
        if min_gini > temp_gini:
            min_gini = temp_gini
            best_sep_val = sep_val
    return min_gini, best_sep_val


def discrete_gini(x, y):
    x, samples = np.unique(x, return_counts=True)
    g = 0
    for value, num in zip(x, samples):
        g += num / x.shape[0] * gini(y[x == value])
    return g, x


def choose_best_feature(X, y, sample_mask, features, feature_type):
    min_gini = 1
    best_feature = None
    best_sep_value = None
    for i in features:
        if feature_type[i] == 'c':
            temp_gini, sep_value = continuous_gini(X[:, i][sample_mask], y[sample_mask])
        else:
            temp_gini, sep_value = discrete_gini(X[:, i][sample_mask], y[sample_mask])
        if min_gini > temp_gini:
            min_gini = temp_gini
            best_feature = i
            best_sep_value = sep_value
    return best_feature, best_sep_value


def build_tree(X, y, sample_mask, features, feature_type):
    # 类别完全相同时停止继续划分
    tree = {}
    c, nums = np.unique(y[sample_mask], return_counts=True)
    if len(c) == 1:
        tree['class'] = c[0]
        tree['is_leaf'] = True
        return tree
    # 遍历完所有特征时， 返回出现次数最多的类别
    if len(features) == 0:
        max_class = nums[0]
        index = 0
        for i in range(len(nums)):
            if max_class < nums[i]:
                max_class = nums[i]
                index = i
        tree["class"] = c[index]
        tree['is_leaf'] = True
        return tree
    best_index, sep_val = choose_best_feature(X, y, sample_mask, features, feature_type)
    tree["index"] = best_index
    tree["sep_val"] = sep_val
    tree["feature_type"] = feature_type[best_index]
    tree["is_leaf"] = False
    new_features = np.copy(features)
    sons = {}
    if feature_type[best_index] == 'd':
        del(new_features[best_index])
        for value in sep_val:
            new_sample_mask = (X[:, best_index] == value) & sample_mask
            sons[value] = build_tree(X, y, new_sample_mask, new_features, feature_type)
    else:
        left_sample_mask = (X[:, best_index] <= sep_val) & sample_mask
        right_sample_mask = (X[:, best_index] > sep_val) & sample_mask
        sons["left"] = build_tree(X, y, left_sample_mask, features, feature_type)
        sons["right"] = build_tree(X, y, right_sample_mask, features, feature_type)
    tree["sons"] = sons
    return tree


def decide_class(x, tree):
    if tree['is_leaf']:
        return tree['class']
    index = tree["index"]
    if tree['feature_type'] == 'c':
        if x[index] <= tree['sep_val']:
            return decide_class(x, tree["sons"]["left"])
        else:
            return decide_class(x, tree["sons"]["right"])
    else:
        return decide_class(x, tree[x[index]])


class DecisionTreeClassifier(object):
    def __init__(self,
                 criterion="gini",
                 max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, is_pruning=False):
        sample_num, col = X.shape
        self.tree = build_tree(X, y,
                               sample_mask=np.ones(sample_num, dtype=bool),
                               features=[i for i in range(col)],
                               feature_type=get_feature_type(X))
    def predict(self, X_test):
        n_samples, cols = X_test.shape
        predict = []
        for i in range(n_samples):
            predict.append(decide_class(X_test[i, :], self.tree))
        return(predict)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return np.sum(y==pred) / X.shape[0]


if __name__ == "__main__":
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    print(dtc.evaluate(X_train, y_train))
    print(dtc.evaluate(X_test, y_test))


