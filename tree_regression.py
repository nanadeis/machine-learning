# 树回归
#2018-6-19
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# from Tkinter import *

class TreeNode():
    def __init__(self, feat, val, right, left):
        self.feature_to_split =feat
        self.value_of_split = val
        self.right_branch =right
        self.left_branch =left


def load_dataset(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat


# 二元切分数据集，数据集每一行是一个样本，每一列是一维特征
def bin_split_dataset(dataset, feature, value):
    mat_left = dataset[np.where(dataset[:,feature] > value)[0], :]
    mat_right = dataset[np.where(dataset[:,feature] <= value)[0], :]
    return mat_left, mat_right

# 回归树
def reg_leaf(dataset):
    return np.mean(dataset[:, -1])


def reg_err(dataset):
    return np.var(dataset[:, -1])*dataset.shape[0]

# 模型树
def linear_solve(dataset):
    X, y = dataset[:, 0:-1], dataset[:, -1]
    b = np.ones((dataset.shape[0], 1))
    X = np.c_[X, b]
    X_T_X = X.T * X
    # if np.linalg.det(X_T_X) == 0:
    #     print("X_T_X不可逆")
    w = np.linalg.pinv(X_T_X) * (X.T * y)
    # w = X_T_X.I * (X.T * y)
    return w, X, y


def model_leaf(dataset):
    w, _, _ = linear_solve(dataset)
    return w

def model_err(dataset):
    w, X, y = linear_solve(dataset)
    y_hat = X*w
    return np.sum(np.power(y - y_hat, 2))

def choose_best_split(dataset, leaf_type, err_type, ops):
    least_err, least_sam = ops[0], ops[1]
    if len(set(dataset[:,-1].T.tolist()[0])) == 1:
        return None, leaf_type(dataset)
    best_feature = 0
    best_value = 0
    min_err = float('inf')
    s = err_type(dataset)
    for feature in range(dataset.shape[1]-1):
        for value in set(np.asarray(dataset[:, feature]).ravel()):
            mat0, mat1 = bin_split_dataset(dataset, feature, value)
            if mat0.shape[0] < least_sam or mat1.shape[0] < least_sam:
                continue
            new_s = err_type(mat0) + err_type(mat1)
            if new_s < min_err:
                best_feature = feature
                best_value = value
                min_err = new_s
    if s - min_err < least_err:
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_dataset(dataset, best_feature, best_value)
    if mat0.shape[0] < least_sam or mat1.shape[0] < least_sam:
        return None, leaf_type(dataset)
    return best_feature, best_value


def creat_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat == None:
        return val
    ret_tree = {}
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    left_set, right_set = bin_split_dataset(dataset, feat, val)
    ret_tree['left'] = creat_tree(left_set, leaf_type, err_type, ops)
    ret_tree['right'] = creat_tree(right_set, leaf_type, err_type, ops)
    return ret_tree

def get_mean(tree):
    if isinstance(tree['left'], dict):
        tree['left'] = get_mean(tree['left'])
    if isinstance(tree['right'], dict):
        tree['right'] = get_mean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0

# 后剪枝
def prune(tree, test_data):
    # 如果没有数据对树进行塌陷处理
    if test_data.shape[0] == 0:
        return get_mean(tree)
    if isinstance(tree['left'], dict) or isinstance(tree['right'], dict):
        l_set, r_set = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
        if isinstance(tree['left'], dict):
            tree['left'] = prune(tree['left'], l_set)
        else:
            tree['right'] = prune(tree['right'], r_set)
    if not isinstance(tree['left'], dict) and not isinstance(tree['right'], dict):
        l_set, r_set = bin_split_dataset(test_data, tree['sp_ind'], tree['sp_val'])
        no_merge_err = np.sum(np.power(l_set[:, -1] - tree['left'], 2)) + np.sum(np.power(r_set[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        merge_err = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if merge_err < no_merge_err:
            print("merge")
            return tree_mean
        else: return tree
    else:
        return tree

def predict(sample, tree, type):
    if not isinstance(tree, dict):
        if type == 'reg':
            return float(tree)
        elif type == 'model':
            return float(sample[:, 0:-1]*tree[0:-1, :] + tree[-1, :])
    ind = tree['sp_ind']
    if sample[:, ind] > tree['sp_val']:
        return predict(sample, tree['left'], type)
    else: return predict(sample, tree['right'], type)


# 决定系数coefficient of determination，记为R2
def evalue(dataset, tree, type = 'reg'):
    s_res = 0
    y_hat = np.ones((dataset.shape[0], 1))
    for i in range(dataset.shape[0]):
        y_hat[i, 0] = predict(dataset[i], tree, type)
        # s_res += (p - dataset[i, -1])**2
    return np.corrcoef(dataset[:, -1], y_hat, rowvar=0)[0, 1]
    # return 1 - s_res/(np.var(dataset[:, -1])*dataset.shape[0])

def isTree(obj):
    return (type(obj).__name__=='dict')

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = inDat.shape[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 0:n] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['sp_ind']] > tree['sp_val']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

if __name__ == "__main__":
    # test = [[1,2,3],[3,2,1], [2,1,3]]
    # test = np.mat(test)
    # tree = creat_tree(test, ops=(0, 1))
    # print(evalue(test, tree))

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, shuffle=True)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    boston = np.mat(np.hstack((X_train, y_train)))
    model_leaf(boston)
    tree = creat_tree(boston, ops=(1, 4))
    print(tree)
    test_data = np.mat(np.hstack((X_test,y_test)))
    print("后剪枝前回归树训练误差：", evalue(boston, tree))
    print("后剪枝前回归树测试误差：", evalue(test_data, tree))
    prune(tree, test_data)
    print(tree)
    print("后剪枝后回归树训练误差：", evalue(boston, tree))
    print("后剪枝后回归树测试误差：", evalue(test_data, tree))
    model_tree = creat_tree(boston, leaf_type=model_leaf, err_type=model_err, ops=(1,4))
    # print(model_tree)
    print("后剪枝前模型树训练误差：", float(evalue(boston, model_tree, 'model')))
    print("后剪枝前模型树测试误差：", float(evalue(test_data, model_tree, 'model')))

    # my_mat = np.mat(load_dataset('E:/workspace/machine_learning/ml_action/exp2.txt'))
    # print(my_mat.shape)
    # model_tree = creat_tree(my_mat, model_leaf, model_err)
    # print(model_tree)
    # bike_train = np.mat(load_dataset('E:/workspace/machine_learning/ml_action/bikeSpeedVsIq_train.txt'))
    # bike_test = np.mat(load_dataset('E:/workspace/machine_learning/ml_action/bikeSpeedVsIq_test.txt'))
    # print(bike_train.shape, bike_test.shape)
    # reg_tree = creat_tree(bike_train, ops=(1, 20))
    # print(evalue(bike_test, reg_tree))
    # y_hat = createForeCast(reg_tree, bike_test[:, 0])
    # print(np.corrcoef(y_hat, bike_test[:, 1], rowvar=0)[0, 1])
    # model_tree = creat_tree(bike_train, model_leaf, model_err, ops=(1, 20))
    # print(evalue(bike_test, model_tree, 'model'))
    # w, X, y = linear_solve(bike_train)
    # print(w)