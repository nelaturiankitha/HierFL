import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier
import time
from collections import Counter

start = time.time()

# Load the data from the CSV file
df = pd.read_csv('IOT_TEST.csv')

print("Original 'type' column values:")
print(df['type'].value_counts())

# Check if 'type' column contains string values
if df['type'].dtype == 'object':
    labelencoder = LabelEncoder()
    df['type'] = labelencoder.fit_transform(df['type'])
else:
    # If 'type' is already numeric, we'll assume it's properly encoded
    print("'type' column is already numeric")

print("\nEncoded 'type' column values:")
print(df['type'].value_counts())

X = df.drop(['ts', 'label', 'type'], axis=1).values
y = df['type'].values

print("\nUnique values in y:")
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1337, stratify=y)

transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

# Decision tree training and prediction
model = DecisionTreeClassifier(random_state=1337)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=4, n_jobs=-1)

print('\nscores:', scores)
print("fit_time: %0.3f " % (scores['fit_time'].mean()))
print("score_time: %0.3f " % (scores['score_time'].mean()))

print("Accuracy (Testing): %0.4f " % (scores['test_accuracy'].mean()))
print("Precision (Testing): %0.4f " % (scores['test_precision_macro'].mean()))
print("Recall (Testing): %0.4f " % (scores['test_recall_macro'].mean()))
print("F1-Score (Testing): %0.4f " % (scores['test_f1_macro'].mean()))

end = time.time()
print("Time taken {}".format(end - start))