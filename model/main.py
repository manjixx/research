from model.models import *
from sklearn.model_selection import train_test_split
from model import utils

if __name__ == '__main__':
    file_path = "../dataset/synthetic_step.csv"
    x_features = ['ta', 'hr']
    c = [1, 0.1, 1]
    w = [1, 1, 1]
    # svm(file_path=file_path, x_features=x_features, index='bmi', c=c)
    # knn_model(file_path=file_path, x_features=x_features, w=w, neighbours=20)
    data = pd.read_csv(file_path, encoding='gbk')
    data = data.dropna(axis=0, how='any', inplace=False)
    data.loc[(data['pmv'] > 0.5), 'pmv'] = 2
    data.loc[(-0.5 <= data['pmv']) & (data['pmv'] <= 0.5), 'pmv'] = 1
    data.loc[(data['pmv'] < -0.5), 'pmv'] = 0

    x = np.array(data[['ta', 'hr']])
    y = np.array(data[['pmv']]).flatten().astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print("start")
    # adaboost(x_train, y_train, x_test, y_test)
    xgboost(x_train, y_train, x_test, y_test)
    # random_forest(x_train, y_train, x_test, y_test)


