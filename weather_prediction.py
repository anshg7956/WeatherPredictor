path = "/content/weatherAUS.csv"
data = pd.read_csv(path)
data.head(10)
# data.drop(['Evaporation','Sunshine'], axis = 1)

data.describe()

data.info()

data.columns

"""## EDA

1) Find the shape of your training and testing datasets (Hint: use the shape() function)
"""

# print(np.sum(data['RainToday'].isna()))
# print(np.sum(data['Pressure3pm'].isna()))
data.head(10)

y = 1 * (data['RainTomorrow'] == 'Yes')

# Your code here
y = 1 * (data['RainTomorrow'] == 'Yes')
X = data[['Pressure3pm','Cloud3pm','Temp3pm']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""2) Plot a histogram of one of the columns you find interesting in the data frame. (Hint: Recall the *.hist()* method)"""

# Your code here
data.hist(bins=50, figsize=(15,10))
plt.show()

"""3) Plot a scatter plot of two columns that you find interesting in the data fraem. (Hint: Recall in *matplotlib.pyplot* there is the *.scatter* method)"""

# Your code here

"""4) Try to think of one more step on your own here. What else would you like to know about the data or how it is arranged?"""

# Your code here

"""## Baseline Model

Fundamentally, a baseline is a model that is both simple to set up and has a reasonable chance of providing decent results.

# To do:

1) Select which variables you will use in your model (start with 2 or 3)

2) Separate training and testing sets

3) Create a linear regression model. Code examples are [here](https://colab.research.google.com/drive/1HC4netVsOZT1BHjyUNcu8u8f1Etfoi2b?authuser=1) for reference.


## Challenge:
- Create another model with different variables to see if you can get higher test accuracy
- Write 3 sentences about why you chose those variables


"""

# Insert your model code here #
data_complete = data[['Pressure3pm','Cloud3pm','Temp3pm','WindGustSpeed','WindSpeed3pm','Humidity3pm','RainTomorrow','MinTemp','MaxTemp']].dropna()
y = 1 * (data_complete['RainTomorrow'] == 'Yes')
X = data_complete[['Pressure3pm','Temp3pm','WindGustSpeed','WindSpeed3pm','Humidity3pm','MinTemp','MaxTemp']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(X_train, y_train)

y_test_proba = logit_model.predict_proba(X_test)[:,1]
y_test_proba

y_test_pred = y_test_proba > 0.515
y_test_pred
y_test_pred = np.multiply(y_test_pred, 1)
y_test_pred
np.mean(y_test == y_test_pred)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test_pred = logit_model.predict(X_test)
np.mean(y_test_pred == y_test)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

"""## Improved Model

For improved models, you can explore more advanced models such as neural network or even recurrent neural network.
"""

# Insert your advanced model code here #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

path = "/content/weatherAUS.csv"
data = pd.read_csv(path)
data.head(10)

num = 0
for column in data.columns:
  num_missing = np.sum(data[column].isna())
  print(f"{column}: {num_missing}")
  num += 1

cities = np.unique(data.Location)
print(cities)

show_evolution('Adelaide','MinTemp')

def show_evolution(city,feature):
  data_city = data[data.Location == city]
  plt.plot(data_city['Date'][:365*2], data_city[feature][:365*2])
  plt.xlabel('Time')
  plt.ylabel(feature)
  plt.show

data_complete = data[['RainTomorrow','Pressure3pm','Humidity3pm','Temp3pm','RainToday','MinTemp','MaxTemp','Location']].dropna()

data_categorical = data_complete[['RainToday','Location']]

dummies = pd.get_dummies(data_categorical)
dummies.head()

data_complete = data_complete.drop(['RainToday','Location'], axis = 1)

data_complete = pd.concat([data_complete, dummies], axis=1)

y = 1 * (data_complete['RainTomorrow'] == 'Yes')

X = data_complete.drop(['RainTomorrow'], axis = 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns = data_complete.drop(['RainTomorrow'], axis = 1).columns)
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(30, activation=tf.nn.relu),
     tf.keras.layers.Dense(15, activation=tf.nn.relu),
     tf.keras.layers.Dense(2, activation = "softmax")
  ])

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, # loss function
    optimizer=tf.keras.optimizers.Adam(), # optimizer function
    metrics=['accuracy'] # reporting metric
)

history = model.fit(
    X_train, y_train, epochs=10,
    validation_data=(X_test, y_test)
)

def modelCompiler(mdel, xTrain, yTrain, xTest, yTest):
  model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy, # loss function
    optimizer=tf.keras.optimizers.Adam(), # optimizer function
    metrics=['accuracy'] # reporting metric
  )
  history = model.fit(
      xTrain, yTrain, epochs=15,
      validation_data = (xTest, yTest)
  )

"""## Tuning

Tuning is the process of maximizing a model's performance without overfitting or creating too high of a variance. In machine learning, this is accomplished by selecting appropriate “hyperparameters.” Hyperparameters can be thought of as the “dials” or “knobs” of a machine learning model.
"""

# Insert your model tuning code here #
model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(15, activation=tf.nn.relu),
     tf.keras.layers.Dense(30, activation=tf.nn.relu),
     tf.keras.layers.Dense(30, activation=tf.nn.relu),
     tf.keras.layers.Dense(15, activation=tf.nn.relu),
     tf.keras.layers.Dense(6, activation=tf.nn.relu),
     tf.keras.layers.Dense(2, activation = "softmax")
  ])

modelCompiler(model, X_train, y_train, X_test, y_test)

accuracyMethod(y_test_pred, model, X_test, y_test)

def accuracyMethod(yTestPred, mdel, xTest, yTest):
  yTestPred = model.predict(xTest)
  print(np.mean(np.argmax(yTestPred, axis = 1) == yTest))

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(["Train Accuracy", "Test Accuracy"])

plt.figure(figsize=(16, 6))
plot_graphs(history, 'accuracy')

model = tf.keras.models.Sequential([
     tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     tf.keras.layers.Dense(4, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     tf.keras.layers.Dense(2, activation = "softmax")
  ])

modelCompiler(model, X_train, y_train, X_test, y_test)
accuracyMethod(y_test_pred, model, X_test, y_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_test_pred = model.predict(X_test)
np.mean(y_test_pred == y_test)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

"""## Performance Summary

Make a presentation of your result. You can refer to the syntax below.

Markdown | Preview
--- | ---
`**Model 1**` | **Model 2**
`*70%*` or `_italicized text_` | *90%*
`` `Monospace` `` | `Monospace`
`~~strikethrough~~` | ~~strikethrough~~
`[A link](https://www.google.com)` | [A link](https://www.google.com)
`![An image](https://www.google.com/images/rss.png)` | ![An image](https://www.google.com/images/rss.png)

More resources about creating tables in markdown of colab can be found [here](https://colab.research.google.com/notebooks/markdown_guide.ipynb#scrollTo=Lhfnlq1Surtk).

## Conclusion

- Does your model (baseline or advanced) answer the questions you are interested in?
- Does your EDA present the visual aid that you need to create the models?
- How can you improve your models if you have one more week of time before submission?
"""