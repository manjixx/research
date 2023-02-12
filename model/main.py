from model.models import *
if __name__ == '__main__':
    file_path = "../dataset/2021.csv"
    x_features = ['ta', 'hr']
    c = [1, 0.1, 1]
    w = [1, 1, 1]
    svm(file_path=file_path, x_features=x_features, index='bmi', c=c)
    # knn_model(file_path=file_path, x_features=x_features, w=w, neighbours=20)
