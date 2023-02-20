from model.models import *

if __name__ == '__main__':
    file_path = "../dataset/2021.csv"
    synthetic = False
    x_features = ['ta', 'hr']
    y_feature = ['thermal sensation']
    c = [1, 0.1, 1]
    w = [1, 1, 1]

    print("算法：pmv")
    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='pmv')
    pmv(np.array(data[x_features]), np.array(data[y_feature]))

    print("算法：svm, 分类指标： bmi")
    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='svm')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='svm', index='bmi',
        x_feature=x_features, y_feature=y_feature,
        normalization=False
    )
    kernel = ['linear', 'poly', 'rbf']
    svm(x_train, y_train, x_test, y_test, kernel, c)

    print("算法：knn, 分类指标： bmi")

    data = read_data(file_path, synthetic=synthetic, season='summer', algorithm='knn')
    x_train, y_train, x_test, y_test = split_svm_knn(
        data, algorithm='knn', index='bmi',
        x_feature=x_features, y_feature=y_feature,
        normalization=False
    )

    distance = ['manhattan', 'manhattan_inverse', 'manhattan_gauss', 'euclid', 'euclid_inverse', 'euclid_gauss']
    knn(x_train, y_train, x_test, y_test, w, distance)

    data = read_data(file_path, synthetic=synthetic, season='summer',
                     algorithm='other')
    weights = [0.5, 1, 0.8]
    x_train, y_train, x_test, y_test, sample_weights, sample_weights_test = split_ensemble(
        data, index='bmi', x_feature=x_features,
        y_feature=y_feature, weights=weights
    )

    print("算法：adaboost，分类指标：bmi")
    adaboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)

    print("算法：random forest，分类指标：bmi")

    random_forest(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)

    print("算法：xgboost，分类指标：bmi")
    xgboost(x_train, y_train, x_test, y_test, sample_weights, sample_weights_test)



