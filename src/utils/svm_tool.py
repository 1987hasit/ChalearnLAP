import sys
import os
import re
from subprocess import *

# svm, grid, and gnuplot executable files

is_win32 = (sys.platform == 'win32')
current_dir = os.getcwd()

if not is_win32:
    raise Exception('not compatible with non-windows system')
else:
    svmtrain_exe = current_dir + "\\libsvm\\windows\\svm-train.exe"
    svmpredict_exe = current_dir + "\\libsvm\\windows\\svm-predict.exe"
    svmscale_exe  = current_dir + "\\libsvm\\windows\\svm-scale.exe"
    grid_py = current_dir + "\\libsvm\\tools\\grid.py"

class SvmTool():
    def __init__(self):
        assert os.path.exists(svmtrain_exe),"svm-train executable not found"
        assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
        assert os.path.exists(grid_py),"grid.py not found"
        
        
    def scale(self,inputfile,scaledfile=None,rangefile=None):
        ''' scale data using libsvm windows tools '''
        if scaledfile == None:
            scaledfile = inputfile + '.scaled'
        if rangefile == None:
            rangefile = inputfile + '.range'
        cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, rangefile, inputfile, scaledfile)
        print('Scaling data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
    
        
    def scalewithrange(self,inputfile,scaledfile,rangefile):
        ''' scale with range '''
        if scaledfile == None:
            scaledfile = inputfile + '.scaled'
        cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, rangefile, inputfile, scaledfile)
        print('Scaling data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
    
        
    def gridsearch(self,trainingfile, kernelType):
        ''' cross validation using grid.py '''
        cmd = ''
        if kernelType == 'RBF':
            cmd = 'python {0} -svmtrain "{1}" "{2}"'.format(grid_py, svmtrain_exe, trainingfile)
        elif kernelType == 'linear':
            # bug here!!!
            cmd = 'python {0} -log2c -1,2,1 -log2g 1,1,1 -t 0 -svmtrain "{1}" "{2}"'.format(grid_py, svmtrain_exe, trainingfile)
        print('Cross validation...')
        f = Popen(cmd, shell = True, stdout = PIPE).stdout
        line = ''
        while True:
            last_line = line
            line = f.readline()
            if not line: break
        c,g,rate = map(float,last_line.split())
        return c,g,rate
    
    def write2SVMFormat(self, outputPath, fileName, X, y):
        ''' write X, y to SVM format 
            y: list of labels [1, -1, 1]
            X: list of data [[1,1,1], [1,2,3], ...]
        '''
        # Create the output
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        problemFile = open(os.path.join(outputPath, fileName), 'w')
        numData = len(y)        # number of data
        dimFeature = len(X[0])  # dimensionality of feature
        
        for i in xrange(numData):
            yi = y[i]
            Xi = X[i]
            
             
            # write to SVM-Format file
            problemFile.write('%d' % yi)
            problemFile.write(' ')

            for j in xrange(dimFeature):
                data = Xi[j]

                problemFile.write('%d' % (j + 1))
                problemFile.write(':')
                problemFile.write('%f' % data)
                problemFile.write(' ')
          
            problemFile.write('\n')  
       
        problemFile.close()
    
    
    def train(self,c,g,trainingfile,modelfile=None):
        ''' train svm using windows tools '''
        if modelfile == None:
            modelfile = trainingfile + '.model'
        cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe,c,g,trainingfile,modelfile)
        print('Training...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()
    
        
    def predict(self,testingfile,modelfile,predictfile = None):
        ''' predict using windows tools '''
        if predictfile == None:
            predictfile = testingfile + '.predict'
        cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, testingfile, modelfile, predictfile)
        print('Testing...')
        f = Popen(cmd, shell = True,stdout= PIPE).stdout
        line = ''
        while True:
            last_line = line
            line = f.readline()
            if not line: break
        accu = re.findall('Accuracy = (100|[0-9]{1,2}|[0-9]{1,2}\.[0-9]+)%.*',last_line)[0]
        return float(accu)

# if __name__ == '__main__':
#     trainingfile = 'splice.txt'
#     rbf = svmRBF()
#     rangefile = trainingfile + '.range'
#     scaledtraining = trainingfile + '.scale'
#     rbf.scale(trainingfile,scaledtraining,rangefile)
# 
#     testingfile = 'splice.t'
#     scaledtesting = testingfile + '.scale'
#     rbf.scalewithrange(testingfile,scaledtesting,rangefile)
#     
#     c,g,r = rbf.gridsearch(scaledtraining)
#     print "best c,g is", c,g
#     modelfile = trainingfile + '.model'
#     rbf.train(c,g,scaledtraining,modelfile)
#     
#     accutest = rbf.predict(scaledtraining,modelfile)
#     accutrain = rbf.predict(scaledtesting,modelfile)
#     print "different on gridsearch, trainingdata and testing is", r, accutrain, accutest