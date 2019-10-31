## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from scipy import optimize

def train(path_to_images, csv_file):
    #path_to_images = "../data/training/images"
    #csv_file = "../data/training/steering_angles.csv"
    #data pre-processing
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    print("data_process")
    #get the human readings
    inter_array=[]
    for i in range (0,frame_nums.shape[0]):
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        resized=cv2.resize(im_full, (int(64), int(60)), interpolation = cv2.INTER_AREA)
        mask=alv_vision(resized, rgb = [-1, 0,1], thresh = 0.85)
        mask1d = mask.ravel()
        mask_norm=mask1d/255
        inter_array.append(mask_norm)
    X=np.asanyarray(inter_array)
    #fil_y=np.asarray(y)
    print("x/y computed")
    #Allocate the steering angles in the bin like ALVINN 
    #dividing the steering angles equally ranging from min to max value
    max_y = np.max(steering_angles,axis=0)
    min_y = np.min(steering_angles,axis=0)
    #y_bins = []
    #for x in range(60):
     #   y_bins.insert(x,[min_y + x*(max_y-min_y)/60])
    y_bins=np.linspace(min_y,max_y,60)
    #y_output = np.asarray(y_bins)
    #y_out = y_output.ravel()
    #print("y_out")
    
    #sort out the bins that matches the steering values and assign the highest value to that bin using gaussian function
    gaussian = [0.1,0.3,0.5,0.7,1.0,0.7,0.5,0.3,0.1]
    gaussian = np.asarray(gaussian)
    final_y = []
    for i in range(0,1500):
        value=abs(np.subtract(y_bins,steering_angles[i]))
        index_min = np.argmin(value)
        #index_min = 0
        if((index_min) > 55): #right most value
            inter_value=(index_min+5)-60
            cut_gauss=gaussian[0:(gaussian.shape[0]-inter_value)]
            binlist = np.zeros(60)
            binlist[index_min-4:index_min+5]=cut_gauss
            #cut_gauss = gaussian[0:gaussian.shape[0]-(60-index_min)]
        elif((index_min) < 4):
            binlist = np.zeros(60)
            #inter_value = abs(index_min-4)
            cut_gauss = gaussian[4-index_min:9]
            #print(len(cut_gauss))
            binlist[0:len(cut_gauss)]=cut_gauss
        else:
            binlist = np.zeros(60)
            binlist[index_min-4:index_min+5]=gaussian
        #print(index_min)
        #print(steering_angles[i])
        final_y.append(binlist)
    y=np.asarray(final_y)
    
    # Train your network here. You'll probably need some weights and gradients!
    NN = NeuralNetwork()
    NN.outputvalueNN=y_bins
    #NN = NeuralNetwork()
    #NN.outputvalue=y_bins
    #yhat =NN.forward(X)
    #print(yhat.shape)
    epoch = 300
    batch_size=40
    lr = 0.5
    for i in range(epoch):
        for batchx,batchy in batch_io(X,y,batch_size):
            grads = NN.computeGradients(batchx,batchy)
            #grads = NN.computeGradients(X,y)
            params = NN.getParams()
            NN.setParams(params - lr*grads)
        #NN.setParams(params)
    print("quitting training")
    #params = NN.getParams()
    #grads = NN.computeGradients(X, y)
    return NN


def alv_vision(image, rgb, thresh):
        '''
        Apply the basic color-based road segmentation algorithm used in 
        the autonomous land vehicle. 
        Args
        image: color input image, dimension (n,m,3)
        rgb: tri-color operation values, dimension (3)
        thresh: threshold value for road segmentation
        Returns
        mask: binary mask of the size (n, m), ones indicate road, zeros indicate non-road    '''
        return (np.dot(image.reshape(-1, 3), rgb) > thresh).reshape(image.shape[0], image.shape[1])
    
def predict(NN, image_file):
    im_full = cv2.imread(image_file)
    resized=cv2.resize(im_full, (int(64), int(60)), interpolation = cv2.INTER_AREA)
    mask=alv_vision(resized, rgb = [-1, 0,1], thresh = 0.85)
    mask1d = mask.ravel()
    mask_norm=mask1d/255
    predict=NN.forward(mask_norm)
    #arg=np.argmax(predict)
    #angle=NN.outputvalueNN[arg]
    angle=NN.outputvalueNN[np.argmax(predict)]
    return angle

def batch_io(X, y,size):
    for i in np.arange(0, X.shape[0], size):
        yield (X[i:i + size], y[i:i + size])
        
class NeuralNetwork(object):
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 3840
        self.outputLayerSize = 60
        self.hiddenLayerSize = 30
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        outputvalueNN = 0
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))