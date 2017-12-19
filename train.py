import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ensemble import *
from sklearn import tree
from PIL import Image
from feature import *
import os

original_data_directory = "./datasets/original/"

face = 'face'
nonface = 'nonface'

def extract(type):
    original_data_path = original_data_directory + type
    count = 0
    for filename in os.listdir(original_data_path):
        #images are converted into a size of 24 * 24 grayscale
        img = Image.open(original_data_path + "/" + filename).convert('L')
        img = img.resize((24, 24))
        img = np.array(img)

        #extract NPD features
        features = []
        feature = NPDFeature(img).extract()
        features.append(feature)

        count = count + 1
        print(count)
    np.save("./datasets/"+type+"_features.npy", features)

if __name__ == "__main__":
    # extract feature
    extract(face)
    extract(nonface)

    face_features = np.load("./datasets/face_features.npy")
    nonface_features = np.load("./datasets/nonface_features.npy")

    num_face_sample, num_face_feature = face_features.shape
    num_nonface_sample, num_nonface_feature = nonface_features.shape

    positive_label = [np.ones(1) for i in range(num_face_sample)]
    negative_label = [-np.ones(1) for i in range(num_nonface_sample)]

    positive_samples = np.concatenate((face_features, positive_label), axis=1)
    negative_samples = np.concatenate((nonface_features, negative_label), axis=1)

    training_size = 800
    rate = 0.5
    data = np.concatenate((positive_samples[:int(training_size*rate), :],
                                 negative_samples[:int(training_size*(1-rate)), :]),
                                axis=0)
    X = data[:training_size, :num_face_feature]
    y = data[:training_size, -1]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=42)
    y_train = y_train.reshape((len(y_train), 1))
    y_validation = y_validation.reshape((len(y_validation), 1))
    print(X_train.shape, y_train.shape, X_validation.shape, y_validation.shape)

    weak_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    classifier = AdaBoostClassifier(weak_classifier, 2)
    classifier.fit(X_train, y_train)

    Y_pred = classifier.predict(X_validation)
    target_names = ['face', 'non face']
    report = classification_report(y_validation, Y_pred, target_names=target_names, digits=4)
    with open("./report.txt", 'w') as f:
        f.write(report)
    print(report)


