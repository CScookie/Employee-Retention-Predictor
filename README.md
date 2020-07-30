# Employee-Retention-Predictor
Predicting if employee will retain in the company using neural network

## Dataset source: 
https://www.kaggle.com/gummulasrikanth/hr-employee-retention
<img src="README_src/HR_csv.PNG" alt="CSV" width=1000>

### There will be 3 notebooks, each having its own purpose as follow:
* Data cleaning
* Data preprocessing
* Model training and testing

# 1. Data cleaning
### Importing in libraries and dataset
```python
import pandas as pd
import numpy as np

raw_data = pd.read_csv('HR_Employee.csv')

raw_data.head()
```
Output:
<img src="README_src/HR_dataframe.PNG" alt="CSV" width=1000>

### Copy dataframe so not to edit the original dataframe
```python
df = raw_data.copy()
```

### Checking info on dataframe. Ensuring there are no missing values
```python
df.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14999 entries, 0 to 14998
Data columns (total 10 columns):
satisfaction_level       14999 non-null float64
last_evaluation          14999 non-null float64
number_project           14999 non-null int64
average_montly_hours     14999 non-null int64
time_spend_company       14999 non-null int64
Work_accident            14999 non-null int64
promotion_last_5years    14999 non-null int64
sales                    14999 non-null object
salary                   14999 non-null object
left                     14999 non-null int64
dtypes: float64(2), int64(6), object(2)
memory usage: 1.1+ MB
```

### Removing unwanted values and encoding categorical datas
```python
df['sales'].unique()
```
Output:
```
array(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'], dtype=object)
```
```python
#dropping rows with value 'RandD'
df = df[df.sales != 'RandD']

df['salary'].unique()
```
Output:
```
array(['low', 'medium', 'high'], dtype=object)
```

```python
#mapping 'low' to 1, 'medium' to 2 and 'high' to 3
df['salary'] = df['salary'].map({'low':1,'medium':2,'high':3})
```

### Get dummies from column 'sales' while dropping the first column

```python
sales_columns = pd.get_dummies(df['sales'], drop_first = True)
df = df.drop(['sales'], axis=1)

df_with_dummies = pd.concat([df, sales_columns], axis=1)
```

### Reorder columns such that the target (column 'left') is the most right

```python
#get all the column names
df_with_dummies.columns.values
```
Output:
```
array(['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'left', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical'], dtype=object)
```

```python
# rearrange column names in the order we want
column_names_reordered = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'accounting', 'hr',
       'management', 'marketing', 'product_mng', 'sales', 'support',
       'technical', 'left']

df_reordered = df_with_dummies[column_names_reordered]
```

### Exporting cleaned dataframe as csv file

```python
df_reordered.to_csv('HR_employee_cleaned.csv',index=False)
```

# 2. Data preprocessing

### Importing libaries and dataset
```python
import numpy as np

# We will use the sklearn preprocessing library, as it will be easier to standardize the data.
from sklearn import preprocessing

# Load the data
raw_csv_data = np.loadtxt('HR_employee_cleaned.csv',delimiter=',', skiprows=1, )


# The inputs are all columns in the csv, [:,0:-1] 
# except the last column, [:,-1] (which is our targets)

unscaled_inputs_all = raw_csv_data[:,0:-1]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:,-1]
```
### Balancing and splitting data into train, validation and test
```python
# Count how many targets are 1 (meaning that the customer did convert)
num_one_targets = int(np.sum(targets_all))

# Set a counter for targets that are 0 (meaning that the customer did not convert)
zero_targets_counter = 0

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# Declare a variable that will do that:
indices_to_remove = []

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

scaled_inputs = unscaled_inputs_equal_priors
```
```python
# When the data was collected it was actually arranged by date
# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.
# Since we will be batching, we want the data to be as randomly spread out as possible
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]
```
```python
# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
```
Output:
```
2775.0 5520 0.5027173913043478
338.0 690 0.48985507246376814
337.0 690 0.48840579710144927
```

### Saving datasets as .npz for tensorflow in the next notebook
```python
# Save the three datasets in *.npz.

np.savez('HR_data_train', inputs=train_inputs, targets=train_targets)
np.savez('HR_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('HR_data_test', inputs=test_inputs, targets=test_targets)
```

# 3. Model training and testing
### Import relevant libaries
```python
# we must import the libraries once again since we haven't imported them in this file
import numpy as np
import tensorflow as tf
```

### Load .npz files and store each in variable
```python
# let's create a temporary variable npz, where we will store each of the three HR datasets
npz = np.load('HR_data_train.npz')

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('HR_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('HR_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
```

### Training the model
Dropout(0.2) and early_stopping used to prevent overfitting of the model

```python
# Set the input and output sizes
input_size = 16
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 20
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dropout(0.2),
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 32

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)

# fit the model
# note that this time the train, validation and test data are not iterable
model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )  
```
Output:
```
Train on 5520 samples, validate on 690 samples
Epoch 1/100
5520/5520 - 1s - loss: 9.7704 - accuracy: 0.4969 - val_loss: 0.7153 - val_accuracy: 0.4246
Epoch 2/100
5520/5520 - 0s - loss: 2.4929 - accuracy: 0.5049 - val_loss: 0.7010 - val_accuracy: 0.5101
Epoch 3/100
5520/5520 - 0s - loss: 1.2742 - accuracy: 0.5130 - val_loss: 0.6861 - val_accuracy: 0.5464
Epoch 4/100
5520/5520 - 0s - loss: 0.9000 - accuracy: 0.5040 - val_loss: 0.6847 - val_accuracy: 0.5493
Epoch 5/100
5520/5520 - 0s - loss: 0.7954 - accuracy: 0.5134 - val_loss: 0.6804 - val_accuracy: 0.6101
Epoch 6/100
5520/5520 - 0s - loss: 0.7468 - accuracy: 0.5317 - val_loss: 0.6824 - val_accuracy: 0.5101
Epoch 7/100
5520/5520 - 0s - loss: 0.7273 - accuracy: 0.5406 - val_loss: 0.6747 - val_accuracy: 0.5913
Epoch 8/100
5520/5520 - 0s - loss: 0.7055 - accuracy: 0.5498 - val_loss: 0.6723 - val_accuracy: 0.7304
Epoch 9/100
5520/5520 - 0s - loss: 0.7060 - accuracy: 0.5524 - val_loss: 0.6683 - val_accuracy: 0.5667
Epoch 10/100
5520/5520 - 0s - loss: 0.6879 - accuracy: 0.5639 - val_loss: 0.6648 - val_accuracy: 0.5464
Epoch 11/100
5520/5520 - 0s - loss: 0.6820 - accuracy: 0.5826 - val_loss: 0.6628 - val_accuracy: 0.5145
Epoch 12/100
5520/5520 - 0s - loss: 0.6640 - accuracy: 0.6013 - val_loss: 0.6502 - val_accuracy: 0.7261
Epoch 13/100
5520/5520 - 0s - loss: 0.6607 - accuracy: 0.6136 - val_loss: 0.6384 - val_accuracy: 0.7435
Epoch 14/100
5520/5520 - 0s - loss: 0.6500 - accuracy: 0.6239 - val_loss: 0.6303 - val_accuracy: 0.7522
Epoch 15/100
5520/5520 - 0s - loss: 0.6394 - accuracy: 0.6351 - val_loss: 0.6221 - val_accuracy: 0.7348
Epoch 16/100
5520/5520 - 0s - loss: 0.6297 - accuracy: 0.6536 - val_loss: 0.6062 - val_accuracy: 0.7783
Epoch 17/100
5520/5520 - 0s - loss: 0.6151 - accuracy: 0.6764 - val_loss: 0.5995 - val_accuracy: 0.8043
Epoch 18/100
5520/5520 - 0s - loss: 0.6132 - accuracy: 0.6605 - val_loss: 0.5753 - val_accuracy: 0.7841
Epoch 19/100
5520/5520 - 0s - loss: 0.5956 - accuracy: 0.6819 - val_loss: 0.5505 - val_accuracy: 0.7928
Epoch 20/100
5520/5520 - 0s - loss: 0.5873 - accuracy: 0.6830 - val_loss: 0.5397 - val_accuracy: 0.8101
Epoch 21/100
5520/5520 - 0s - loss: 0.5740 - accuracy: 0.7065 - val_loss: 0.5249 - val_accuracy: 0.8246
Epoch 22/100
5520/5520 - 0s - loss: 0.5593 - accuracy: 0.7071 - val_loss: 0.5249 - val_accuracy: 0.8217
Epoch 23/100
5520/5520 - 0s - loss: 0.5538 - accuracy: 0.7150 - val_loss: 0.4914 - val_accuracy: 0.8420
Epoch 24/100
5520/5520 - 0s - loss: 0.5500 - accuracy: 0.7149 - val_loss: 0.4845 - val_accuracy: 0.8391
Epoch 25/100
5520/5520 - 0s - loss: 0.5444 - accuracy: 0.7118 - val_loss: 0.4690 - val_accuracy: 0.8420
Epoch 26/100
5520/5520 - 0s - loss: 0.5420 - accuracy: 0.7234 - val_loss: 0.4529 - val_accuracy: 0.8507
Epoch 27/100
5520/5520 - 0s - loss: 0.5407 - accuracy: 0.7246 - val_loss: 0.4687 - val_accuracy: 0.8493
Epoch 28/100
5520/5520 - 0s - loss: 0.5282 - accuracy: 0.7351 - val_loss: 0.4515 - val_accuracy: 0.8507
Epoch 29/100
5520/5520 - 0s - loss: 0.5219 - accuracy: 0.7446 - val_loss: 0.4448 - val_accuracy: 0.8565
Epoch 30/100
5520/5520 - 0s - loss: 0.5233 - accuracy: 0.7353 - val_loss: 0.4255 - val_accuracy: 0.8522
Epoch 31/100
5520/5520 - 0s - loss: 0.5175 - accuracy: 0.7444 - val_loss: 0.4403 - val_accuracy: 0.8522
Epoch 32/100
5520/5520 - 0s - loss: 0.5174 - accuracy: 0.7504 - val_loss: 0.4311 - val_accuracy: 0.8580
Epoch 33/100
5520/5520 - 0s - loss: 0.4929 - accuracy: 0.7716 - val_loss: 0.4020 - val_accuracy: 0.8580
Epoch 34/100
5520/5520 - 0s - loss: 0.4962 - accuracy: 0.7708 - val_loss: 0.4019 - val_accuracy: 0.8638
Epoch 35/100
5520/5520 - 0s - loss: 0.4904 - accuracy: 0.7726 - val_loss: 0.4011 - val_accuracy: 0.8652
Epoch 36/100
5520/5520 - 0s - loss: 0.4881 - accuracy: 0.7812 - val_loss: 0.3853 - val_accuracy: 0.8681
Epoch 37/100
5520/5520 - 0s - loss: 0.4842 - accuracy: 0.7835 - val_loss: 0.4025 - val_accuracy: 0.8522
Epoch 38/100
5520/5520 - 0s - loss: 0.4789 - accuracy: 0.7846 - val_loss: 0.3847 - val_accuracy: 0.8623
Epoch 39/100
5520/5520 - 0s - loss: 0.4707 - accuracy: 0.7857 - val_loss: 0.3654 - val_accuracy: 0.8725
Epoch 40/100
5520/5520 - 0s - loss: 0.4666 - accuracy: 0.7893 - val_loss: 0.3782 - val_accuracy: 0.8739
Epoch 41/100
5520/5520 - 0s - loss: 0.4651 - accuracy: 0.7924 - val_loss: 0.3763 - val_accuracy: 0.8696
Epoch 42/100
5520/5520 - 0s - loss: 0.4630 - accuracy: 0.7918 - val_loss: 0.3621 - val_accuracy: 0.8696
Epoch 43/100
5520/5520 - 0s - loss: 0.4587 - accuracy: 0.7933 - val_loss: 0.3564 - val_accuracy: 0.8710
Epoch 44/100
5520/5520 - 0s - loss: 0.4539 - accuracy: 0.7973 - val_loss: 0.3564 - val_accuracy: 0.8739
Epoch 45/100
5520/5520 - 0s - loss: 0.4578 - accuracy: 0.7953 - val_loss: 0.3639 - val_accuracy: 0.8855
Epoch 46/100
5520/5520 - 0s - loss: 0.4493 - accuracy: 0.8116 - val_loss: 0.3604 - val_accuracy: 0.8783
Epoch 47/100
5520/5520 - 0s - loss: 0.4456 - accuracy: 0.8132 - val_loss: 0.3565 - val_accuracy: 0.8725
```

### Model testing
```python
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
```
Output:
```
690/690 [==============================] - 0s 20us/sample - loss: 0.3964 - accuracy: 0.8667
```
```python
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
```
Output:
```
Test loss: 0.40. Test accuracy: 86.67%
```
