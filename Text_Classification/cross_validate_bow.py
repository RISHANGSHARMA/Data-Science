import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Read data
df = pd.read_csv('data_set_200.csv')

# Shuffle data
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

print("Different authors in the data-set ", df.Author.unique())

print("Different genre in the data-set ", df['Genre'].unique())

print("Total number of sentences ", len(df['Sentence']))


X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Author'], train_size=0.7, random_state=0)


# BOW
count_vect = CountVectorizer(ngram_range=(1, 3), min_df=5, lowercase=True, stop_words='english')

full_X = count_vect.fit_transform(df['Sentence'])
print(full_X.shape)

X_train_counts = count_vect.fit_transform(X_train)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


clf_NB = MultinomialNB().fit(X_train_counts, y_train)

clf_SVM = SGDClassifier().fit(X_train_counts, y_train)

clf_DT = DecisionTreeClassifier().fit(X_train_counts, y_train)

clf_KNN = KNeighborsClassifier().fit(X_train_counts, y_train)


# Example sentence by Author
# sen = "Vanity and pride are different things, though the words are often used synonymously. A person may be proud " \
#       "without being vain. Pride relates more to our opinion of ourselves, vanity to what we would have others think " \
#       "of us. "

# functions to test sentences

def predictAuthorNB(x):
    return print(clf_NB.predict(count_vect.transform([x])))


def predictAuthorKNN(x):
    return print(clf_KNN.predict(count_vect.transform([x])))


def predictAuthorSVM(x):
    return print(clf_SVM.predict(count_vect.transform([x])))


def predictAuthorDT(x):
    return print(clf_DT.predict(count_vect.transform([x])))


X_test_counts = count_vect.transform(X_test)

predictions_NB = clf_NB.predict(X_test_counts)
print(accuracy_score(y_test, predictions_NB))
predictions_KNN = clf_KNN.predict(X_test_counts)
print(accuracy_score(y_test, predictions_KNN))
predictions_SVM = clf_SVM.predict(X_test_counts)
print(accuracy_score(y_test, predictions_SVM))
predictions_DT = clf_DT.predict(X_test_counts)
print(accuracy_score(y_test, predictions_DT))

# CLASSIFICATION REPORT

print("Classification report for Decision Tree")
print(classification_report(y_test, predictions_DT, target_names=df.Author.unique()))

print("Classification report for Decision Tree")
print(classification_report(y_test, predictions_SVM, target_names=df.Author.unique()))

print("Classification report for Decision Tree")
print(classification_report(y_test, predictions_NB, target_names=df.Author.unique()))

print("Classification report for Decision Tree")
print(classification_report(y_test, predictions_KNN, target_names=df.Author.unique()))

# TEN FOLD CROSS VALIDATION

print("\nTen-Fold Cross-Validation\n")

scores = cross_val_score(clf_NB, full_X, df['Author'], cv=10)
print("Mean Score NB", scores.mean())
scores = cross_val_score(clf_DT, full_X, df['Author'], cv=10)
print("Mean Score DT", scores.mean())
scores = cross_val_score(clf_KNN, full_X, df['Author'], cv=10)
print("Mean Score KNN", scores.mean())
scores = cross_val_score(clf_SVM, full_X, df['Author'], cv=10)
print("Mean Score SVM", scores.mean())


models = [MultinomialNB(), SGDClassifier(), KNeighborsClassifier(), DecisionTreeClassifier()]
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, full_X, df.Author, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


