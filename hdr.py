'''
This is a Neural Network script that performs handwritten digit recognition
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image,ImageEnhance, ImageOps


def initialize_wmatrices(n_layers):

	weights_matrices = [] 
	for i in range(1,len(n_layers)):

		rows =  n_layers[i]
		cols = n_layers[i-1]
		epsilon = math.sqrt(6)/math.sqrt(rows+cols) # From Exercise 4 Courser (Andrew Ng)
		weights_matrices += [np.random.uniform(low=-epsilon, high=epsilon, size=(rows,cols))]

	return weights_matrices


def sigmoid(x):

	return 1.0/(1.0+np.exp(-x))


def sigmoid_der(x):

	return sigmoid(x)*(1-sigmoid(x))

def import_data(inputfile):

	#data = pd.read_csv(inputfile)
	data = pd.read_parquet(inputfile)

	return data

def draw_number(array_digit):
	'''
	Plots a digit given a 28x28 array that represents a handwritten digit.
	'''

	grid = array_digit.reshape((28,28))

	plt.imshow(grid,cmap=plt.get_cmap('gray'))
	plt.show()


def feed_forward(input_layer,weights_matrices):

	a_list = [] # list of hidden layers (without aplying sigmoid)
	for i in range(0,len(weights_matrices)):

		hiden = weights_matrices[i].dot(input_layer)
		a_list+=[hiden]
		input_layer = sigmoid(hiden)


	return input_layer,a_list # Returns the output layer (despite it is called input_layer) and the a_list


def backprop(output_layer,a_list,groud_truth,weights_matrices,input_layer,alpha):
	# https://sudeepraja.github.io/Neural/

	#### Update last W
	a_last = a_list[-1]

	delta_last = np.multiply((output_layer-groud_truth),sigmoid_der(a_last))[np.newaxis].T
	x_last =  sigmoid(a_list[-2])[np.newaxis]
	gradient_last = delta_last.dot(x_last)
	weights_matrices[-1]-=alpha*gradient_last # update weights
	delta = delta_last


	#### Update intermediate Ws
	len_hiden_middle = len(a_list)-2
	for i in range(len_hiden_middle,0,-1):

		wtd = weights_matrices[i+1].T.dot(delta)    # W transpose times delta
		fwx = sigmoid_der(a_list[i])[np.newaxis].T  # f(Wx)
		delta_new = np.multiply(wtd,fwx)
		x = a_list[i-1][np.newaxis]
		gradient = delta_new.dot(x)
		weights_matrices[i]-=alpha*gradient

		delta = delta_new
	
	#### update W0
	wtd = weights_matrices[1].T.dot(delta)    	# W transpose times delta
	fwx = sigmoid_der(a_list[0])[np.newaxis].T  # f(Wx)
	delta_new = np.multiply(wtd,fwx)
	x = input_layer[np.newaxis]  
	gradient = delta_new.dot(x)
	weights_matrices[0]-=alpha*gradient	

	return weights_matrices


def vectorize_number(number):
	'''
	Turns an integer number into a vector. eg: 0 -> [1,0,...,0]
	'''
	out = np.zeros(10)
	out[number] = 1

	return out 

def train_weights(dataset,weights_matrices,number_examples,alpha):

	for i in range(0,number_examples):
		input_layer = np.array(dataset.iloc[i].tolist()[1:])
		ground = int(dataset.iloc[i].tolist()[0])
		vectorized_ground = vectorize_number(ground)
		output_layer,a_list = feed_forward(input_layer,weights_matrices)
		weights_matrices = backprop(output_layer,a_list,vectorized_ground,weights_matrices,input_layer,alpha)


	return weights_matrices


def calculate_confusion_matrix(data_test,weights_matrices,ni,nf):

	confusion_matrix = np.zeros((10, 10))

	for i in range(ni,nf+1):
		input_layer = np.array(data_test.iloc[i].tolist()[1:])
		ground = int(data_test.iloc[i].tolist()[0])
		#draw_number(input_layer)
		output_layer,a_list = feed_forward(input_layer,weights_matrices)
		prediction = output_layer.argmax()

		#print("ground:",ground)
		#print("Predicción:",prediction)
		confusion_matrix[ground,prediction]+=1

	return confusion_matrix

def performance_metrix(confusion_matrix):

	accuracy = confusion_matrix.trace()/confusion_matrix.sum()
	precision = 0
	recall = 0

	for i in range(0,10):
		precision += confusion_matrix[i,i]/np.sum(confusion_matrix[:,i])
		recall+= confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

	precision=precision/10.0
	recall=recall/10.0
	F1score = (2*precision*recall)/(precision+recall)

	return accuracy,precision,recall,F1score

def calculate_performance_vs_trainingsize(trainingsize_list,n_layers,alpha,data,data_test):

	accuracy_list = []
	precision_list = []
	recall_list = []
	F1score_list = []
	weights_matrices = 0


	for n in trainingsize_list: 
		del weights_matrices  # just in case it brings information from the past
		weights_matrices = initialize_wmatrices(n_layers)
		weights_matrices = train_weights(data,weights_matrices,n,alpha)
		confusion_matrix = calculate_confusion_matrix(data_test,weights_matrices,1,500)
		accuracy,precision,recall,F1score = performance_metrix(confusion_matrix)

		accuracy_list += [accuracy]
		precision_list += [precision]
		recall_list += [recall]
		F1score_list += [F1score]

	return accuracy_list,precision_list,recall_list,F1score_list

########################  MAIN  ########################
def main():

	################ PARAMETERS ################
	n_layers = [784,200,100,10] 	# Each element is the number of neurons in each layer
	inputfile = 'data/mnist_train.parquet'
	inputfile_test = 'data/mnist_test.parquet'
	alpha = 0.001
	############################################

	data_train = import_data(inputfile)
	data_train = data_train.sample(frac=1).reset_index(drop=True) # Shuffle data rows
	data_test = import_data(inputfile_test)
	data_test = data_train.sample(frac=1).reset_index(drop=True)  # Shuffle data rows

	trainingsize_list = [100,1000,3000,5000,10000,20000,30000,40000,50000]
	accuracy_list,precision_list,recall_list,F1score_list = calculate_performance_vs_trainingsize(trainingsize_list,n_layers,alpha,data_train,data_test)


	dfoutput = pd.DataFrame({'0.training_size':trainingsize_list,
		'1.accuracy':np.round(accuracy_list,3),'2.precision':np.round(precision_list,3),
		'3.Recall':np.round(recall_list,3),
		'4.F1score_list':np.round(F1score_list,3)})
	dfoutput.to_csv('output2.csv', index=False,sep='\t')

	#print(accuracy_list)

	#weights_matrices = initialize_wmatrices(n_layers)
	#weights_matrices = train_weights(data,weights_matrices,25000,alpha)
	#confusion_matrix = calculate_confusion_matrix(data_test,weights_matrices,10,200)
	#accuracy,precision,recall,F1score = performance_metrix(confusion_matrix)

	##################
	#input_layer = np.array(data.iloc[1].tolist()[1:])/256.0
	#ground = int(data.iloc[1].tolist()[0])
	#vectorized_ground = vectorize_number(ground)
	#output_layer,a_list = feed_forward(input_layer,weights_matrices)
	#weights_matrices = backprop(output_layer,a_list,vectorized_ground,weights_matrices,input_layer,alpha)
	##################	


	#print("DOWN:\n",weights_matrices)
	
	#input_layer = np.array(data_test.iloc[10].tolist()[1:])
	#gray_img = cv2.imread('9_pinta_28x28.png', cv2.IMREAD_GRAYSCALE)
	#gray_img = cv2.bitwise_not(gray_img)
	#draw_number(gray_img)
	#gray_img = (gray_img.reshape((784,)))
	#print(gray_img)
	
	#output_layer,a_list = feed_forward(gray_img,weights_matrices)
	#print("Predicción (grund=9):",output_layer.argmax())
	#print(output_layer)

if __name__=='__main__':
     main()