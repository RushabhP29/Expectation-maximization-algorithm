import math
import numpy as np
from scipy.stats import multivariate_normal
from pyspark import SparkConf, SparkContext
from operator import add

class EMN:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None

    def copyArray(self, array):
        new_arr = [None] * len(array)
        for i in range(len(array)):
            new_arr[i]=  array[i]
        return new_arr


    def closestCluster(self, p, centers):
        bestIndex = 0
        minDist = float("+inf")  # minimum distance
        for i in range(len(centers)):
            tempDist = np.sum((p - centers[i]) ** 2)  # **: exponentiation
        if tempDist < minDist:
            minDist = tempDist
        bestIndex = i
        return bestIndex

    def assigningLabels(self, csv, filename):
        parsedData = csv.map(self.parseLine)
        trueValue = csv.map(self.getTrueValue)

        DELTA = np.eye(54) * 1e-7

        for i in range(2):
            if (np.linalg.det(self.sigma[i]) == 0):  # compute determinant of matrix
                self.sigma[i] = self.sigma[i] + DELTA  # to avoid singular matrix

        predictedProb = parsedData.map(lambda point: self.weightedPDF(point, 2, self.theta, self.mu  , self.sigma))

        result = predictedProb.map(lambda line: self.calc_greatest_index(line))
        results=result.collect()
        true = trueValue.collect()
        accuracy_count = 0  # count how many data points having correct labels
        # # output in results.txt: i-th row: true label, predicted label for i-th data point:
        with open(filename, "w") as f:
            f.write("true\tpredicted\n")
            for i in range(len(results)):
                f.write(str(true[i]) + "\t" + str(results[i]) + "\n")
                if int(true[i]) == int(results[i]):
                    accuracy_count += 1
            accuracy = accuracy_count / len(results)
            if accuracy < 0.5:  # our predicted label IDs might be opposite
                accuracy = 1 - accuracy
        print("accuracy from EM is :", accuracy)


    def calc_greatest_index(self, arr):
        if arr[0]> arr[1]:
            return 0
        elif arr[1]> arr[0]:
            return 1


    def sumError(self,array0, array1):
        return sum([math.fabs(x) for x in (array0 - array1)])

    def getTrueValue(self,line):
        y = np.array([float(x) for x in line.split(',')])
        return y[-1] # return the last element (index -1 means last index)

    def parseLine(self,line):
        y = np.array([float(x) for x in line.split(',')])
        return y[:-1] # get elements starting from index 0 through last-1

    def convert(self,arr):
        size = len(arr)
        size = math.sqrt(size)
        return np.reshape(arr, (int(size), int(size)))

    def calc_tranpose(self,point, probability):
        features = len(point)
        result = np.zeros(features * features)
        index = 0
        for i in range(features):
            for j in range(features):
                square = point[i] * point[j] * probability
                result[index] = square
                index += 1
        return result

    def weightedPDF(self,point, K, theta, mu, sigma):
        pdf = np.zeros(K) # initialize array of size K
        for i in range(K):
            pdf[i] = self.gaussianPDF(point, mu[i], sigma[i]) * theta[i]
        return pdf/sum(pdf)

    def sub_point_mu(self,point, mu):
        return [a - b for a, b in zip(point, mu)]

    def gaussianPDF(self, point, mu, sigma):
        return multivariate_normal(mu, sigma).pdf(point)

    def normPDF(self,point, mu, sigma):
        return multivariate_normal(mu, sigma).pdf(point)

    def elementWiseAdd(self, list1, list2):
        return [a + b for a, b in zip(list1, list2)]


    def EMClustering(self, csv, K, maxIteration):

        parsedData = csv.map(self.parseLine)
        trueValue = csv.map(self.getTrueValue)

        features = len(parsedData.take(1)[0])  # number of features of a data point
        print("number of features: ", features)

        samples = parsedData.count()  # number of data points
        print("number of data points: ", samples)  # print for debugging

        # intialize three model parameters:
        self.theta = np.ones(K) / 2  # theta_1, theta_2 = 0.5

        self.mu = parsedData.takeSample(False, K, 1)  # mu[0], mu[1], ... mu[K-1]

        self.sigma = np.zeros((K, features, features))

        K=2

        for i in range(K):
            self.sigma[i] = np.eye(features)  # sigma[0] is co-variance of points in c0

        if (maxIteration < 2):
            maxIteration = 50

        DELTA = np.eye(features) * 1e-7  # fixed diagonal matrix of small values, to avoid singular matrix

        for count in range(maxIteration):
            for i in range(K):
                if (np.linalg.det(self.sigma[i]) == 0):  # compute determinant of matrix
                    #print("enter here")
                    self.sigma[i] = self.sigma[i] + DELTA  # to avoid singular matrix

            error = 0.0
            oldTheta = self.copyArray(self.theta)

            thetaPDF = parsedData.map(lambda point: (point, self.theta[0] * self.normPDF(point, self.mu[0], self.sigma[0]), self.theta[1] * self.normPDF(point, self.mu[1], self.sigma[1])))

            pointProb0 = thetaPDF.map(lambda line: (line[0], line[1] / (line[1] + line[2])))
            pointProb1 = thetaPDF.map(lambda line: (line[0], line[2] / (line[1] + line[2])))

            self.theta[0] = pointProb0.map(lambda line: line[1]).reduce(add)  # need to divide by samples later
            self.theta[1] = pointProb1.map(lambda line: line[1]).reduce(add)

            self.theta = self.theta/ samples

            self.mu[0] = pointProb0.map(lambda line: line[0] * line[1]).reduce(self.elementWiseAdd)
            self.mu[0] = self.mu[0] / (samples * self.theta[0])

            self.mu[1] = pointProb1.map(lambda line: line[0] * line[1]).reduce(self.elementWiseAdd)
            self.mu[1] = self.mu[1] / (samples * self.theta[1])

            sigma0= pointProb0.map(lambda line : self.calc_tranpose(self.sub_point_mu(line[0], self.mu[0]), line[1])).reduce(self.elementWiseAdd)
            sigma0 = sigma0/(samples * self.theta[0])

            sigma1 = pointProb1.map(lambda line: self.calc_tranpose(self.sub_point_mu(line[0], self.mu[0]), line[1])).reduce(self.elementWiseAdd)
            sigma1 = sigma1 / (samples * self.theta[1])

            self.sigma[0] = self.convert(sigma0)
            self.sigma[1] = self.convert(sigma1)
            error += self.sumError(oldTheta, self.theta)

            if (error < 1e-7):
                 break


