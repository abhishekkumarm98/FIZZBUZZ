import numpy as np              # It defines a fast numerical array and matrix structure as it has binding with C-libraries.
import pandas as pd             # It allows for fast analysis and high performance of data.
import tensorflow as tf         # It is a deep learning framework and used for numerical computation using data flow graphs.
from keras.utils import np_utils    # It is also a deep learning framework and run over tensorflow for fast experimentation.
from tqdm import tqdm_notebook      # It is used for showing progressbar, progressmeter, progress, meter, rate and time etc.
import matplotlib.pyplot as plt     # It is a plotting package, used for plotting quality 2D and 3D graphs.
%matplotlib inline                  


# Logic Based fizzbuzz function [Software1.0]
#Logic Based Explanation

def fizzBuzz(i):

    if i % 15 == 0:
        return "FizzBuzz"
    elif i % 3 == 0:
        return "Fizz"
    elif i % 5 == 0:
        return "Buzz"
    else:
        return "Other"


# Create Training and Testing Datasets in CSV Format
def createInputCSV(start,end,filename):
    
    inputData   = []
    outputData  = []
    
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzBuzz(i))

    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename) # To create a file, from dictionary to CSV format.
    
    print(filename, "Created!")
 
 
# Processing Input and Label Data
def processData(dataset):
    
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel
    
def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # The reason to take number 10 in the range because our training data set starts from 101 and ends at 1000. 
        # So to cover upto 1000th number we need 10 bits.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel),4)  # Having four categories in output which will be
                                                                # converted into binary class matrix.
                                                                
                                                                
# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)

# Tensorflow Model Definition

# Defining Placeholder
inputTensor  = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])

NUM_HIDDEN_NEURONS_LAYER_1 = 128

LEARNING_RATE = 0.03

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# Initializing the input to hidden layer weights
input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])

# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the first hidden layer
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))

# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)

# Training the Model
epochs = [i*1000 for i in range(1,8)]
BATCH_SIZE = 128
training_accuracy = []

for NUM_OF_EPOCHS in epochs:
  tra_acc=[]
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):
        
        #Shuffle the Training Dataset at each epoch
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData  = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]
        
        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end], 
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        tra_acc.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                             sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                             outputTensor: processedTrainingLabel})))
    training_accuracy.append(sum(tra_acc)/len(tra_acc)*100)
        
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})
    
plt.plot(epochs, training_accuracy)
plt.title("Number of epochs Vs. Accuracy")
plt.xlabel("Number of epochs ------>")
plt.ylabel("Accuracy ------>")

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"
        
# Testing the Model [Software 2.0]

total_correct = 0
total_incorrect = 0

correct_fizz = correct_Buzz = correct_fizzBuzz = correct_other = 0
Incorrect_fizz = Incorrect_Buzz = Incorrect_fizzBuzz = Incorrect_other = 0

predictedTestLabelList = []

for i,j in zip(processedTestingLabel,predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))
    
    if np.argmax(i) == j:
        total_correct = total_correct + 1
        
        if np.argmax(i) == 1:
            correct_fizz +=1
        elif np.argmax(i) == 2:
            correct_Buzz +=1
        elif np.argmax(i) ==3:
            correct_fizzBuzz +=1
        else:
            correct_other += 1
            
    else:
        total_incorrect = total_incorrect + 1
        if np.argmax(i) == 1:
            Incorrect_fizz +=1
        elif np.argmax(i) == 2:
            Incorrect_Buzz +=1
        elif np.argmax(i) == 3:
            Incorrect_fizzBuzz +=1
        else:
            Incorrect_other += 1

print("Total incorrect: " + str(total_incorrect), " Total Correct :" + str(total_correct))
print("Testing Accuracy: " + str(total_correct/(total_correct + total_incorrect)*100)+"\n")

print("Incorrect: " + str(Incorrect_fizz), " Correct :" + str(correct_fizz))
print("Testing Accuracy of fizz: " + str(correct_fizz/(Incorrect_fizz + correct_fizz)*100)+"\n")

print("Incorrect: " + str(Incorrect_Buzz), " Correct :" + str(correct_Buzz))
print("Testing Accuracy of Buzz: " + str(correct_Buzz/(Incorrect_Buzz + correct_Buzz)*100)+"\n")

print("Incorrect: " + str(Incorrect_fizzBuzz), " Correct :" + str(correct_fizzBuzz))
print("Testing Accuracy of fizzBuzz: " + str(correct_fizzBuzz/(Incorrect_fizzBuzz + correct_fizzBuzz)*100)+"\n")

print("Incorrect: " + str(Incorrect_other), " Correct :" + str(correct_other))
print("Testing Accuracy of other: " + str(correct_other/(Incorrect_other + correct_other)*100))

testingData["predicted_label"] = predictedTestLabelList
testingData.to_csv("output.csv")
