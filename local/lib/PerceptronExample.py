from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def sigmoide(u):
	g = np.exp(u)/(1 + np.exp(u))
	return g
def Plot_Perceptron():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X2 = X[:100][:,:2]
	y2 = y[:100]
	fig, (ax0, ax1) = plt.subplots(1,2)
	ax0.scatter(X2[:,0], X2[:,1], c=y2, cmap="Accent");

	#Aprendizaje
	MaxIter = 100000
	w = np.ones(3).reshape(3, 1)
	eta = 0.001
	N = len(y2)
	Error =np.zeros(MaxIter)
	Xent = np.concatenate((X2,np.ones((100,1))),axis=1)
	for i in range(MaxIter):
		tem = np.dot(Xent,w)
		tem2 = sigmoide(tem.T)-np.array(y2)
		Error[i] = np.sum(abs(tem2))/N
		tem = np.dot(Xent.T,tem2.T)
		wsig = w - eta*tem/N
		w = wsig
	print("Weights:")
	print(w)
	print('Error=',Error[-1])
	#Grafica de la frontera encontrada
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	X2 = X[:100][:,:2]
	y2 = y[:100]
	ax1.scatter(X2[:,0], X2[:,1], c=y2,cmap="Accent");
	x1 = np.linspace(4,8,20)
	x2 = -(w[0]/w[1])*x1 - (w[2]/w[1])
	ax1.plot(x1,x2,'k')