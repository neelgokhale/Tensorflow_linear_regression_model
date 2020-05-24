# Tensorflow_linear_regression_model
Basic outline of linear regression neural network to predict values from randomized datasets

## Importing & Setup


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
import numpy as np
import pandas as pd
```


```python
# load test dataset

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

```




    (sex                          male
     age                            22
     n_siblings_spouses              1
     parch                           0
     fare                         7.25
     class                       Third
     deck                      unknown
     embark_town           Southampton
     alone                           n
     Name: 0, dtype: object,
     0)



## Visualizing


```python
dftrain.age.hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b95de21288>




![png](/img/output_5_1.png)



```python
dftrain['sex'].value_counts().plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b95e0cbe88>




![png](/img/output_6_1.png)



```python
dftrain['class'].value_counts().plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b95e0d9fc8>




![png](/img/output_7_1.png)



```python
pd.concat([dftrain, y_train], axis=1).groupby('sex')['survived'].mean().plot(kind='barh').set_xlabel('% survive')
```




    Text(0.5, 0, '% survive')




![png](output_8_1.png)


## Categorizing & Numerating Dataset


```python
dftrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>age</th>
      <th>n_siblings_spouses</th>
      <th>parch</th>
      <th>fare</th>
      <th>class</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>Third</td>
      <td>unknown</td>
      <td>Southampton</td>
      <td>n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>First</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>Third</td>
      <td>unknown</td>
      <td>Southampton</td>
      <td>y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>First</td>
      <td>C</td>
      <td>Southampton</td>
      <td>n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Third</td>
      <td>unknown</td>
      <td>Queenstown</td>
      <td>y</td>
    </tr>
  </tbody>
</table>
</div>




```python
CATEGORICAL_COLUMNS = ['sex', 'class', 'deck', 'embark_town', 'parch', 'alone',  'n_siblings_spouses']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

feature_columns
```




    [VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'), dtype=tf.string, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='class', vocabulary_list=('Third', 'First', 'Second'), dtype=tf.string, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='deck', vocabulary_list=('unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E'), dtype=tf.string, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='embark_town', vocabulary_list=('Southampton', 'Cherbourg', 'Queenstown', 'unknown'), dtype=tf.string, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='parch', vocabulary_list=(0, 1, 2, 5, 3, 4), dtype=tf.int64, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='alone', vocabulary_list=('n', 'y'), dtype=tf.string, default_value=-1, num_oov_buckets=0),
     VocabularyListCategoricalColumn(key='n_siblings_spouses', vocabulary_list=(1, 0, 3, 4, 2, 5, 8), dtype=tf.int64, default_value=-1, num_oov_buckets=0),
     NumericColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     NumericColumn(key='fare', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]



## Input Function & Training


```python
# function(

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
clear_output()
print(result['accuracy'])
print(result)
```

    0.75
    {'accuracy': 0.75, 'accuracy_baseline': 0.625, 'auc': 0.8319253, 'auc_precision_recall': 0.78191656, 'average_loss': 0.4997142, 'label/mean': 0.375, 'loss': 0.49567127, 'precision': 0.64102566, 'prediction/mean': 0.4409824, 'recall': 0.75757575, 'global_step': 200}
    

## Prediction


```python
result = list(linear_est.predict(eval_input_fn))

print(len(result))

clear_output()

index = int(input("Enter index number: "))
print(dfeval.loc[index])


print("( Actual, Prediction ) --> (", y_eval.loc[index], ",", result[index]['probabilities'][1], ")")

```

    Enter index number:  5
    

    sex                       female
    age                           15
    n_siblings_spouses             0
    parch                          0
    fare                      8.0292
    class                      Third
    deck                     unknown
    embark_town           Queenstown
    alone                          y
    Name: 5, dtype: object
    ( Actual, Prediction ) --> ( 1 , 0.761648 )
    

## Visualizing Error


```python
# Visualizing deviation

pred_list = []
ind_list = []
deviation = []
lrg_dev = np.zeros(len(result))

for i in range(len(result)):
    pred_list.append(result[i]['probabilities'][1])
    ind_list.append(i)
    deviation.append(y_eval[i] - pred_list[i])
    if abs(deviation[i]) > 0.25:
        lrg_dev[i] = deviation[i]

plt.bar(ind_list, deviation, color='g')
plt.bar(ind_list, lrg_dev, color='r')
```




    <BarContainer object of 264 artists>




![png](/img/output_17_1.png)

