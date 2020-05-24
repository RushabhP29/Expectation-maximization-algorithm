
from pyspark import SparkConf, SparkContext
from EMN import EMN
from Kmeans import Kmeans

def main():
    sc = SparkContext(master="local", appName="EM")
    try:
        csv = sc.textFile("kmeans_data.csv") #csv =sc.textFile(sys.argv[1]) if input via cmd
    except IOError:
        print('No such file')
        exit(1)
    K=2
    maxIteration = 2
    myEM = EMN()
    myEM.EMClustering(csv, K, maxIteration)
    outfile = "EMresults.txt"
    myEM.assigningLabels(csv, outfile)
    sc.stop()


if __name__ == "__main__":
    main()

