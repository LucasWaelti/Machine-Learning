import numpy as np
import math

from TrainingDataGenerator import *

global ERROR

class Neuron:
	def __init__(self,func='tanh',num_weights=0):
		self.lamda = 1
		self.theta = 0
		self.activationFunction = None
		self.functionType = func
		self.weights = []
		self.bias = 0
		self.num_weights = num_weights
		self.net = 0
		self.output = 0
		self.delta = 0
		self.setActivationFunction(func)
		self.initWeights()

	def initWeights(self):
		for _ in range(0,self.num_weights):
			self.weights.append(np.random.random())
		self.bias = np.random.random()

	def setActFuncParam(self,lamda=1,theta=0):
		self.lamda = lamda
		self.theta = theta

	def linear(self,net,d=0):
		if(not d):
			return self.lamda*(net-self.theta)
		else:
			return self.lamda
	def tanh(self,net,d=0):
		if(not d):
			return math.tanh(self.lamda*(net-self.theta))
		else:
			return self.lamda*(1 - math.tanh(self.lamda*(net-self.theta))**2)
	def sigmoid(self,net,d=0):
		arg = -self.lamda*(net-self.theta)
		arg = math.exp(arg)
		out = 1/(1 + arg)
		if(not d):
			return out
		else:
			# Condensed form of the derivative
			return self.lamda*out*(1-out) 
	def relu(self,net,d=0):
		if(net < 0):
			return 0
		else:
			if(not d):
				return self.lamda*(net-self.theta)
			else:
				return self.lamda
	def leaky_relu(self,net,d=0, leak = 0.1):
		if(net < 0):
			if(not d):
				return leak*self.lamda*(net-self.theta)
			else:
				return leak*self.lamda
		else:
			if(not d):
				return self.lamda*(net-self.theta)
			else:
				return self.lamda

	def setActivationFunction(self,func='tanh'):
		if(func == 'linear'):
			self.activationFunction = self.linear
		elif(func == 'tanh'):
			self.activationFunction = self.tanh
		elif(func == 'sigmoid'):
			self.activationFunction = self.sigmoid
		elif(func == 'relu'):
			self.activationFunction = self.relu
		elif(func == 'leaky_relu'):
			self.activationFunction = self.leaky_relu
		else:
			print('Error: invalid activation function specified')
			exit() # Exit program, network is compromised

	def eval(self,inputs):
		# inputs is a list containing the output of each neuron from previous layer
		net = 0
		if(self.num_weights == 1):
			net += self.weights[0]*inputs[0]
		else:
			for i in range(0,self.num_weights):
				net += self.weights[i]*inputs[i]
		self.net = net + self.bias
		self.output = self.activationFunction(self.net)



class FCCN: # fullyConnectedConvolutionnalNeuralNetwork
	def __init__( self,net_info={'InputDim':1,'HL':2,'in':[1,'tanh'],1:[5,'tanh'],2:[3,'tanh'],'out':[1,'linear']} ):
		# InputDim defines the number of weights the neurons of the input layer have. 
		# HL : Hidden Layer -> there are accessible trough their respective number 1 to n hidden layers
		self.net_info = net_info
		self.layers_index = 0 + self.net_info['HL'] # For adressing each layer from 0 to layers_index
		if(self.net_info['out'][0]): # If there is an output layer
			self.layers_index += 1
		self.neurons = [[]]
		self.eta = 0.001
		self.errorThreshold = 0.005
		self.allowedIter = 100000

		# Build input layer
		for _ in range(0,self.net_info['in'][0]):
			self.neurons[0].append(Neuron(self.net_info['in'][1],self.net_info['InputDim']))

		# If neural net has hidden layers, build them
		if(self.net_info['HL'] > 0): 
			for h in range(1,self.net_info['HL']+1):
				self.neurons.append([]) # Append new empty layer
				for _ in range(0,self.net_info[h][0]):
					# Add neurons to layer h
					if(h == 1):
						# Need to query info from input layer
						self.neurons[h].append(Neuron(self.net_info[h][1],self.net_info['in'][0]))
					else:
						# Query info from previous hidde layer otherwise
						self.neurons[h].append(Neuron(self.net_info[h][1],self.net_info[h-1][0]))

		# If neural net has an output layer, build it
		if(self.net_info['out'][0] > 0):
			self.neurons.append([]) # Append output layer
			last_hidden = self.net_info['HL']
			if(last_hidden == 0):
				# In case there are no hidden layers
				last_hidden = 'in'
			for _ in range(0,self.net_info['out'][0]):
				# Add neurons to layer
				self.neurons[self.net_info['HL']+1].append(Neuron(self.net_info['out'][1],self.net_info[last_hidden][0]))

		self.error = [0 for _ in range(0,self.net_info['out'][0])]
		return

	def setThresIterEta(self,thres=0.005,max_iter=100000,eta=0.001):
		self.errorThreshold = thres
		self.allowedIter = max_iter

	def showNetworkDetails(self):
		print('Global network info:')
		print(self.net_info)

		for l in range(0,self.layers_index+1):
			if(l==0):
				print('Input layer:')
			elif(l==self.layers_index):
				print('Output layer:')
			else:
				print('Hidden layer '+str(l)+':')
			layer_size = len(self.neurons[l])
			for n in range(0,layer_size):
				print(str(self.neurons[l][n])+' '+str(n)+' '+str(self.neurons[l][n].functionType))
		print(' ')
		return

	def getOutputFromLayer(self,layer,size):
		output = []
		for i in range(0,size):
			output.append(self.neurons[layer][i].output)
		return output

	def forward(self,t_input):
		data_size = len(t_input)
		if(data_size != self.net_info['in'][0]):
			print("FCCN.forward(): input data not adapted to network's input layer")
			exit()
		else:
			# Feed input layer
			layer_size = len(self.neurons[0])
			for n in range(0,layer_size):
				self.neurons[0][n].eval(t_input[n])
			# Propagate in network
			for l in range(1,self.layers_index+1):
				layer_size = len(self.neurons[l])
				output = self.getOutputFromLayer(l-1,len(self.neurons[l-1]))
				for n in range(0,layer_size):
					self.neurons[l][n].eval(output)

	def calculateDeltas(self,t_out=0):
		# Ouptut layer
		layer = self.net_info['HL']+1
		if(t_out is not 0): # There is an output to compare
			# Solve output layer
			for i in range(0,self.net_info['out'][0]): 
				delta = 2*(self.neurons[layer][i].output - t_out[i])
				delta *= -self.neurons[layer][i].activationFunction(self.neurons[layer][i].net,d=True)
				self.neurons[layer][i].delta = delta 


			# Solve rest of the layers
			rem_layers = self.net_info['HL'] # In + hidden (count from 0)
			for i in range(0,rem_layers+1): 
			# For each layer
				ind = rem_layers - i
				num_current = len(self.neurons[ind]) # Number of neurons in current layer
				for j in range(0,num_current): 
				# For each neuron of the layer
					delta = self.neurons[ind][j].activationFunction(self.neurons[ind][j].net,d=True)
					num_posterior = len(self.neurons[ind+1]) # Number of neurons in the posterior layer
					s = 0
					for k in range(0,num_posterior): 
					# Get all deltas of each neuron of the posterior layer
						s += self.neurons[ind+1][k].delta * self.neurons[ind+1][k].weights[j]
					self.neurons[ind][j].delta = delta * s
		else:
			print('Error: No true output given to be compared.')
			exit()

	def update_weights(self,t_inp):
		for i in range(0,self.net_info['HL']+2):
		# For all layers 
			for j in range(0,len(self.neurons[i])):
			# For each neuron
				for k in range(0,len(self.neurons[i][j].weights)):
				# For each weight
					if(i == 0):
						self.neurons[i][j].weights[k] += self.eta * self.neurons[i][j].delta * t_inp[j][k]
					else:
						self.neurons[i][j].weights[k] += self.eta * self.neurons[i][j].delta * self.neurons[i-1][k].output
				self.neurons[i][j].bias += self.eta * self.neurons[i][j].delta
		return

	def computeAverageError(self,t_output):
		error = 0
		numOutNeurons = self.net_info['out'][0]
		for i in range(0,numOutNeurons):
			error += t_output[i] - self.neurons[self.net_info['HL']+1][i].output 
		error /= numOutNeurons
		return error

	def train(self, t_input, t_output):
		global ERROR

		p = len(t_input)
		error = 1000.0
		counter = 0
		iteration = 0

		while(error > self.errorThreshold and iteration < self.allowedIter):
			iteration += 1
			for i in range(0,p):
				self.forward(t_input[i])
				self.calculateDeltas(t_output[i])
				self.update_weights(t_input[i])
				error = self.computeAverageError(t_output[i])
				ERROR = error 
			
			counter += 1
			if(counter >= 1000):
				print('Average error: ' + str(error), end=' ')
				print('at iteration ' + str(iteration) + '.')
				counter = 0

		if(error < self.errorThreshold):
			print('The error is' +  'smaller than the tolerance. Learning is done.')
			print('It took ' + str(iteration) + ' iterations.')
			output = self.getOutputFromLayer(2,1)
		if(iteration >= self.allowedIter):
			print('The network did not converge fast enough. Process aborted.')
		return


def main():
	# Specify network's dimensions and specifications
	# net_info={'InputDim':2,'HL':3,'in':[8,'sigmoid'],1:[10,'tanh'],2:[6,'relu'],3:[3,'tanh'],'out':[2,'linear']}
	# t_input = [[0.1,0.1],[0.2,0.1],[0.3,0.1],[0.4,0.1],[0.5,0.1],[0.6,0.1],[0.7,0.1],[0.8,0.1]]

	# Network for interpolation from R1 -> R1 (1|2|1 layers)
	net_info = {'InputDim':1,'HL':1,'in':[1,'tanh'],1:[2,'tanh'],'out':[1,'linear']}
	
	t_input,t_output = generateData()

	nn = FCCN(net_info)
	nn.setThresIterEta(thres=0.005,max_iter=100000,eta=0.001)
	nn.showNetworkDetails()
	nn.train(t_input,t_output)

if __name__ == "__main__":
    main()