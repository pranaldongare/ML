import os
from typing import Any, Union
import numpy as np
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.preprocessing import MinMaxScaler
import time

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from CSV file
testing_df_initial = pd.read_csv("train.csv")
testing_df_initial = testing_df_initial.drop(testing_df_initial.columns[[0,3,8]],axis=1)

#Let us drop Cabin column as it has too many non values and it doesn't add to the value of our model.
#Let us also manipulate Age column so as to mention age 28 for non values
#Let us also remove two rows which have no data for Embarked
#The rationale behind above is already explained when this model was created using Sklearn library
print (list(testing_df_initial.columns))

testing_df_initial = testing_df_initial.drop(columns='Cabin', axis=1)

#Now let us replace the Ages. We have dropped Cabin in above call.
Age = testing_df_initial['Age'].median()
for index, row in testing_df_initial.iterrows():
    if np.isnan(row['Age']):
        testing_df_initial.at[index,'Age'] = Age

#Let us validate if Age column is modified correctly
print(testing_df_initial.isnull().sum())

#With above print statement we validate that Age don't contain any non values now.
#Embarked has couple of non values, we can safely drop those rows as number is less
testing_df_initial = testing_df_initial.dropna()
#Print shape and null count of training_df_initial to ensure everything is right
print(testing_df_initial.isnull().sum())
print(testing_df_initial.shape)

#We need to create binary variables for different categorical variables so that we can feed this to the regressor

#Create a list of unique values in column Pclass[])
print(list(testing_df_initial.columns))
Pclass_list = list(testing_df_initial['Pclass'].unique())

#Create new columns with is_Pclass_xx name to be added to testing_df_initial
for member in Pclass_list:
    testing_df_initial['is_Pclass_{}'.format(member)]=0
    for index, row in testing_df_initial.iterrows():
        if row['Pclass']==member:
            row['is_Pclass_{}'.format(member)]=1

#Create a list of unique values in column Sex
Sex_list = list(testing_df_initial['Sex'].unique())

#Create new columns with is_Sex_xx name to be added to testing_df_initial
for member in Sex_list:
    testing_df_initial['is_Sex_{}'.format(member)]=0
    for index, row in testing_df_initial.iterrows():
        if row['Sex']==member:
            row['is_Sex_{}'.format(member)]=1

#Create a list of unique values in column Embarked
Embarked_list = list(testing_df_initial['Embarked'].unique())

#Create new columns with is_Embarked_xx name to be added to testing_df_initial
for member in Embarked_list:
    testing_df_initial['is_Embarked_{}'.format(member)]=0
    for index, row in testing_df_initial.iterrows():
        if row['Embarked']==member:
            row['is_Embarked_{}'.format(member)]=1

#Now we will remove the columns which we have seperated above
testing_df_initial = testing_df_initial.drop(testing_df_initial.columns[[1,2,7]],axis=1)

#Now randomize the data set as we will need to split it later on to get training and testing data seperated
testing_df_initial = testing_df_initial.reindex(np.random.permutation(testing_df_initial.index))

#Create training dataset for regression
training_dataset = testing_df_initial.head(624)

#Dataset below will be used as input variable
training_label_df = training_dataset.drop(columns='Survived',axis=1).values

#Dataset below will be used as output variable
training_feature_df = training_dataset[['Survived']].values

#Create testing dataset for validation
testing_dataset = testing_df_initial.tail(267)

#Dataset below will be used as testing input variable
testing_label_df = testing_dataset.drop(columns='Survived',axis=1).values

#Dataset below will be used as testing output variable
testing_feature_df = testing_dataset[['Survived']].values

#We need to scale the input functions within 0 to 1 range so that neural network can perform properly
Feature_scaler = MinMaxScaler(feature_range=(0, 1))
Label_scaler = MinMaxScaler(feature_range=(0, 1))

#Scale the training datasets
training_label_df_scaled = Feature_scaler.fit_transform(training_label_df)
training_feature_df_scaled = Label_scaler.fit_transform(training_feature_df)

#Scale the testing datasets
testing_label_df_scaled = Feature_scaler.transform(testing_label_df)
testing_feature_df_scaled = Label_scaler.transform(testing_feature_df)


# Define model parameters
learning_rate = 0.01
training_epochs = 100
display_step = 10

# Define how many inputs and outputs are in our neural network
number_of_inputs = testing_label_df.shape[1]
number_of_outputs = 1

#Define input features to be passed to linear_classifier_regressor
#input_features = set(training_dataset)

#Define tuple to be passed to train function
#def get_input_tuple():
#    input_tuple = (training_label_df,training_feature_df)
#    return input_tuple

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 1000
layer_2_nodes = 1000
layer_3_nodes = 1000

# Define the layers of the neural network itself


# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32,shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.round(tf.sigmoid(tf.matmul(X, weights) + biases))

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.round(tf.sigmoid(tf.matmul(layer_1_output, weights) + biases))

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.round(tf.sigmoid(tf.matmul(layer_2_output, weights) + biases))

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    output = tf.matmul(layer_3_output, weights) + biases
    prediction = tf.round(tf.sigmoid(output))

# Section Two: Define the cost function of the neural network that will be optimized during training
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
#with tf.variable_scope('logging'):
    #tf.summary.scalar('current_cost', cost)
    #summary = tf.summary.merge_all()

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Create log file writers to record training progress.
    # We'll store training and testing log data separately.
    #training_writer = tf.summary.FileWriter('./logs/training', session.graph)
    #testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: training_label_df_scaled, Y: training_feature_df_scaled})

        # Every few training steps, log our progress
        if epoch % display_step == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            #training_cost, training_summary = session.run([cost, summary], feed_dict={X: training_label_df_scaled, Y: training_feature_df_scaled})
            #testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: testing_label_df_scaled, Y: testing_feature_df_scaled})
            training_cost = session.run(cost, feed_dict={X: training_label_df_scaled, Y: training_feature_df_scaled})
            testing_cost = session.run(cost, feed_dict={X: testing_label_df_scaled, Y: testing_feature_df_scaled})
            # Write the current training status to the log files (Which we can view with TensorBoard)
            ##training_writer.add_summary(training_summary, epoch)
            ##testing_writer.add_summary(testing_summary, epoch)

            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {:5f}  Testing Cost: {:5f}".format(epoch, training_cost, testing_cost))

    # Training is now complete!

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: training_label_df_scaled, Y: training_feature_df_scaled})
    final_testing_cost = session.run(cost, feed_dict={X: testing_label_df_scaled, Y: testing_feature_df_scaled})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the "prediciton" operation
    testing_predicted_scaled = session.run(prediction, feed_dict={X: testing_label_df_scaled})

    # Unscale the data back to it's original units
    testing_predicted = Label_scaler.inverse_transform(testing_predicted_scaled)

    #Try to print predicted and actual data for testing dataset for first 5 values
    loop_count = 0
    Correct_Prediction = 0
    Wrong_Prection = 0
    #Count_of_one_Real = 0
    #Count_of_one_Predicted = 0
    while loop_count < 267:
        Real_Survivor = testing_dataset['Survived'].values[loop_count]
        Predicted_Survivor = testing_predicted[loop_count][0]
        loop_count = loop_count+1
        #if Real_Survivor==1:
            #Count_of_one_Real = Count_of_one_Real+1
        #if Predicted_Survivor==1:
            #Count_of_one_Predicted = Count_of_one_Predicted+1
        if Real_Survivor==Predicted_Survivor:
            Correct_Prediction = Correct_Prediction+1
        else:
            Wrong_Prection = Wrong_Prection+1

    ##real_earnings = test_data_df['total_earnings'].values[0]
    ##predicted_earnings = Y_predicted[0][0]
        #print("The person in actual was {}".format(Real_Survivor))
        #print("Prediction is{}".format(Predicted_Survivor))
    print ("Correct = {}, Wrong ={}".format(Correct_Prediction,Wrong_Prection))