import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
#split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def create_datasetMultipleTimesBackAhead(dataset, n_steps_out=1, n_steps_in = 1, overlap = 1):
	dataX, dataY = [], []
	tem = n_steps_in + n_steps_out - overlap
	for i in range(int((len(dataset) - tem)/overlap)):
		startx = i*overlap
		endx = startx + n_steps_in
		starty = endx
		endy = endx + n_steps_out
		a = dataset[startx:endx, 0]
		dataX.append(a)
		dataY.append(dataset[starty:endy, 0])
	return np.array(dataX), np.array(dataY)

def PintaResultado(dataset,trainPredict,testPredict,look_back):
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset+1)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
	#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	#testPredictPlot[len(dataset)-len(testPredict):len(dataset)+1, :] = testPredict
	# plot baseline and predictions
	plt.plot(dataset,label='Original Time serie')
	plt.plot(trainPredictPlot,label='Training prediction')
	plt.plot(testPredictPlot,label='Test prediction')
	plt.legend()
	plt.show()


def EstimaRMSE(model,X_train,X_test,y_train,y_test,scaler,look_back):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_RNN(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],n_steps,look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],n_steps,look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_MultiStep(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = []
	for i in range(X_test.shape[0]):
		temPredict = np.zeros([n_steps])
		for j in range(n_steps):
			if j==0:
				xtest = X_test[i,:]
			else:
				xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
			temPredict[j] = model.predict(xtest.reshape(1,look_back))
		testPredict.append(temPredict)
	testPredict = np.array(testPredict)
	testPredict = testPredict.flatten()
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_MultiOuput(model,X_train,X_test,y_train,y_test,scaler,look_back):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back))
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back))
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.flatten().reshape(-1, 1))
	trainY = scaler.inverse_transform([y_train.flatten()])
	testPredict = scaler.inverse_transform(testPredict.flatten().reshape(-1, 1))
	testY = scaler.inverse_transform([y_test.flatten()])
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_RNN_MultiStep(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps,flag):
	# make predictions
	if flag == 1:#multiple times set as features
		trainPredict = model.predict(X_train.reshape(X_train.shape[0],1,look_back))
		testPredict = []
		for i in range(X_test.shape[0]):
			temPredict = np.zeros([n_steps])
			for j in range(n_steps):
				if j==0:
					xtest = X_test[i,:]
				else:
					xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
				temPredict[j] = model.predict(xtest.reshape(1,1,look_back))
			testPredict.append(temPredict)
		testPredict = np.array(testPredict)
		testPredict = testPredict.flatten()
	else: #multiple times set as times
		trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back,1))
		testPredict = []
		for i in range(X_test.shape[0]):
			temPredict = np.zeros([n_steps])
			for j in range(n_steps):
				if j==0:
					xtest = X_test[i,:]
				else:
					xtest = np.concatenate((X_test[i,j:],temPredict[:j]))
				temPredict[j] = model.predict(xtest.reshape(1,look_back,1))
			testPredict.append(temPredict)
		testPredict = np.array(testPredict)
		testPredict = testPredict.flatten()

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
	trainY = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY.reshape(-1, 1), testPredict.reshape(-1, 1)))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict

def EstimaRMSE_RNN_MultiStepEncoDeco(model,X_train,X_test,y_train,y_test,scaler,look_back,n_steps):
	# make predictions
	trainPredict = model.predict(X_train.reshape(X_train.shape[0],look_back,1))
	trainPredict = trainPredict.flatten()
	testPredict = model.predict(X_test.reshape(X_test.shape[0],look_back,1))
	testPredict = testPredict.flatten()
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
	trainY = scaler.inverse_transform(y_train.flatten().reshape(-1, 1))
	testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
	testY = scaler.inverse_transform(y_test.flatten().reshape(-1, 1))
	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY.flatten().reshape(-1, 1), trainPredict.reshape(-1, 1)))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY.flatten().reshape(-1, 1), testPredict.reshape(-1, 1)))
	print('Test Score: %.2f RMSE' % (testScore))
	return trainPredict, testPredict