import tensorflow as tf
from sklearn.model_selection import train_test_split
from titanic import clean_data, get_features_labels
import pandas as pd

SEX = 'Sex'
AGE = 'Age'
CLASS = 'Pclass'
FARE = 'Fare'
AGE_MISSING = 'AgeMissing'
SURVIVED = 'Survived'


def learn(train_set, test_set):
    x_train, y_train = get_features_labels(train_set)
    x_test, y_test = get_features_labels(test_set)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                activation_fn=tf.nn.softplus,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2)
    classifier.fit(x=x_train,
                   y=y_train,
                   steps=100)

    accuracy_score = classifier.evaluate(x=x_test,
                                         y=y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))

def main():
    data = clean_data("train.csv")
    print('data-size-total:', data.shape[0])
    train_set, test_set = train_test_split(data, random_state=1)

    learn(train_set, test_set)


if __name__ == "__main__":
    main()
    # compute_validation()
