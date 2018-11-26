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
tdf = pd.read_csv("train.csv")
#Create a list of unique values in column X0
X0_list = list(tdf['X0'].unique())
#Create new columns with is_X0_xx name to be added to tdf
for member in X0_list:
    tdf['is_X0_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X0']==member:
            row['is_X0_{}'.format(member)]=1

#Create a list of unique values in column X1
X1_list = list(tdf['X1'].unique())
#Create new columns with is_X1_xx name to be added to tdf
for member in X1_list:
    tdf['is_X1_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X1']==member:
            row['is_X1_{}'.format(member)]=1

#Create a list of unique values in column X2
X2_list = list(tdf['X2'].unique())
#Create new columns with is_X2_xx name to be added to tdf
for member in X2_list:
    tdf['is_X2_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X2']==member:
            row['is_X2_{}'.format(member)]=1

#Create a list of unique values in column X3
X3_list = list(tdf['X3'].unique())
#Create new columns with is_X3_xx name to be added to tdf
for member in X3_list:
    tdf['is_X3_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X3']==member:
            row['is_X3_{}'.format(member)]=1

#Create a list of unique values in column X4
X4_list = list(tdf['X4'].unique())
#Create new columns with is_X4_xx name to be added to tdf
for member in X4_list:
    tdf['is_X4_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X4']==member:
            row['is_X4_{}'.format(member)]=1

#Create a list of unique values in column X5
X5_list = list(tdf['X5'].unique())
#Create new columns with is_X5_xx name to be added to tdf
for member in X5_list:
    tdf['is_X5_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X5']==member:
            row['is_X5_{}'.format(member)]=1

#Create a list of unique values in column X6
X6_list = list(tdf['X6'].unique())
#Create new columns with is_X6_xx name to be added to tdf
for member in X6_list:
    tdf['is_X6_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X6']==member:
            row['is_X6_{}'.format(member)]=1

#Create a list of unique values in column X8
X8_list = list(tdf['X8'].unique())
#Create new columns with is_X8_xx name to be added to tdf
for member in X8_list:
    tdf['is_X8_{}'.format(member)]=0
    for index, row in tdf.iterrows():
        if row['X8']==member:
            row['is_X8_{}'.format(member)]=1

#Drop columns which we don't think can add any value to regression model
tdf_to_use = tdf.drop(tdf.columns[[0,2,3,4,5,6,7,8,9]],axis=1)
#Now randomize the data set as we will need to split it later on to get training and testing data seperated
tdf_to_use = tdf_to_use.reindex(np.random.permutation(tdf_to_use.index))

#Create training numpy array for regression
tds_temp = tdf_to_use.head(2946)
#tds below will be used as input variable
tds = tds_temp.drop(columns='y', axis=1).values
#tds_label below will be used as output variable
tds_label = tds_temp[['y']].values

#Load testing data set from CSV file
testing_temp = tdf_to_use.tail(1262)
testing_df = testing_temp.drop(columns='y',axis=1).values
testing_label = testing_temp[['y']].values

#We need to scale the data to 0 to 1 in order for DNN to work properly.
Y_scaler = MinMaxScaler(feature_range=(0, 1))
#Scale tds_label as it is in integer and need to fit in the range. All other parameters are boolean
tds_label_scaled = Y_scaler.fit_transform(tds_label)

#We need to scale training output data as well with same scale
testing_label_scaled = Y_scaler.transform(testing_label)

# Define model parameters
learning_rate = 0.001
training_epochs = 150
display_step = 30

# Define how many inputs and outputs are in our neural network
number_of_inputs = tds.shape[1]
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 500
layer_2_nodes = 500
layer_3_nodes = 500

# Section One: Define the layers of the neural network itself
# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32,shape=(None, number_of_inputs))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# Section Two: Define the cost function of the neural network that will be optimized during training
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

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
        session.run(optimizer, feed_dict={X: tds, Y: tds_label_scaled})

        # Every few training steps, log our progress
        if epoch % display_step == 0:
            # Get the current accuracy scores by running the "cost" operation on the training and test data sets
            training_cost, training_summary = session.run([cost, summary], feed_dict={X: tds, Y:tds_label_scaled})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={X: testing_df, Y:testing_label_scaled})

            # Write the current training status to the log files (Which we can view with TensorBoard)
            ##training_writer.add_summary(training_summary, epoch)
            ##testing_writer.add_summary(testing_summary, epoch)

            # Print the current training status to the screen
            print("Epoch: {} - Training Cost: {}  Testing Cost: {}".format(epoch, training_cost, testing_cost))

    # Training is now complete!

    # Get the final accuracy scores by running the "cost" operation on the training and test data sets
    final_training_cost = session.run(cost, feed_dict={X: tds, Y: tds_label_scaled})
    final_testing_cost = session.run(cost, feed_dict={X: testing_df, Y: testing_label_scaled})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the "prediciton" operation
    testing_predicted_scaled = session.run(prediction, feed_dict={X: tds})

    # Unscale the data back to it's original units
    testing_predicted = Y_scaler.inverse_transform(testing_predicted_scaled)

    #Try to print predicted and actual data for testing dataset for first 5 values
    loop_count = 0
    while loop_count < 5:
        real_testing_time = testing_temp['y'].values[loop_count]
        predicted_testing_time = testing_predicted[loop_count][0]
        loop_count = loop_count+1
    ##real_earnings = test_data_df['total_earnings'].values[0]
    ##predicted_earnings = Y_predicted[0][0]
        print("The actual testing time was {}".format(real_testing_time))
        print("Predicted time is{}".format(predicted_testing_time))