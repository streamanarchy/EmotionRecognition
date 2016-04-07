import glob
import numpy as np
from scipy.io import wavfile
from voiceFeature import VoiceFeature
from voiceInput import VoiceInput
from neuralNetwork import Trainer
from sklearn import preprocessing as pr
import pickle

class Training():
    def __init__(self,neuralnetwork):
        self.trainer = {}
        self.success = 0
        self.failure = 0
        self.maxlist = [-999]*34
        self.minlist = [999]*34
        self.denom = []
        self.countlist = [0,0,0,0,0,0,0]
        self.Fs = 16000
        self.neuralnetwork = neuralnetwork
        self.trainer['W'] = Trainer(self.neuralnetwork['W'])
        self.trainer['L'] = Trainer(self.neuralnetwork['L'])
        self.trainer['E'] = Trainer(self.neuralnetwork['E'])
        self.trainer['A'] = Trainer(self.neuralnetwork['A'])
        self.trainer['F'] = Trainer(self.neuralnetwork['F'])
        self.trainer['T'] = Trainer(self.neuralnetwork['T'])
        self.trainer['N'] = Trainer(self.neuralnetwork['N'])
        self.voicefeature = VoiceFeature()
        self.emolist = []
        self.voiceinput  = VoiceInput()

    def emo_db_complete_processing(self,feature_train,emotion='N'):
        count = 0
        fileaddress = "/home/project/Documents/Project/training/wav/*.wav"
        fileList = glob.glob(fileaddress)

        for fileindex in xrange(0,fileList.__len__()):
            trainWav = fileList[fileindex]
            """for trainWav in fileList:
            l,x = wavfile.read(trainWav)

            segmentsArray,probOnSet = self.voiceinput.audioSVMSegmentation(xraw,0.02,0.02,plot = False)

            x = np.array([])
            for segments in segmentsArray:
                if x.shape<=0:
                    x = xraw[segments[0]*self.Fs:segments[1]*self.Fs]
                    continue
                else:
                    x = np.hstack((x,xraw[segments[0]*self.Fs:segments[1]*self.Fs]))

            features = self.voicefeature.FeatureExtraction(x,x.shape[0],x.shape[0])"""

            result = trainWav[-6]
            print trainWav
            print result
            """
            if result == 'F':
                y = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                if self.countlist[0] >= 58:
                    continue
                self.countlist[0] = self.countlist[0]+1
            elif result =='W':
                y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                if self.countlist[1] >= 70:
                    continue
                self.countlist[1] = self.countlist[1]+1
            elif result =='A':
                y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

                if self.countlist[2] >= 60:
                    continue
                self.countlist[2] = self.countlist[2]+1
            elif result =='L':
                y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

                if self.countlist[3] >= 50:
                    continue
                self.countlist[3] = self.countlist[3]+1

            elif result =='E':
                y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

                if self.countlist[4] >= 60:
                    continue
                self.countlist[4] = self.countlist[4]+1

            elif result =='T':
                y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

                if self.countlist[5] >= 60:
                    continue
                self.countlist[5] = self.countlist[5]+1
            elif result =='N':
                y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

                if self.countlist[6] >= 60:
                    continue
                self.countlist[6] = self.countlist[6]+1"""

            """features = features.T.tolist()
            print features.__len__()
            for axis0 in xrange(0,features.__len__()):
                normalizedFeatures = []
                for axis1 in xrange(0,features[axis0].__len__()):
                    normalizedFeatures.append((features[axis0][axis1]-self.minlist[axis1])/self.denom[axis1])
                print normalizedFeatures
                for res,emoTrainer in self.trainer.items():
                    if res == result:
                        self.neuralnetwork[res]=emoTrainer.train(normalizedFeatures,1.0)
                        print "Same:",res,result
                    else:
                        self.neuralnetwork[res] = emoTrainer.train(normalizedFeatures,0.0)
                        print "Not Same",res,result"""

            """if result == 'L':
                self.neuralnetwork['L'] = self.trainer['L'].train(feature_train[fileindex],1.0)
            else :
                self.neuralnetwork['L'] = self.trainer['L'].train(feature_train[fileindex],0.0)"""

            for res,emoTrainer in self.trainer.items():
                if res == result:
                    self.neuralnetwork[res]=emoTrainer.train(feature_train[fileindex],1.0)
                    print "Same:",res,result
                else:
                    self.neuralnetwork[res] = emoTrainer.train(feature_train[fileindex],0.0)
                    print "Not Same",res,result

        return self.neuralnetwork

    def feature_normalisation(self):
        count =0
        fileList = glob.glob("/home/project/Documents/Project/training/wav/*.wav")
        featuresarray = np.array([])
        resultarray = []
        for trainWav in fileList:
            l,x = wavfile.read(trainWav)
            result = trainWav[-6]
            features = self.voicefeature.FeatureExtraction(x,x.shape[0],x.shape[0])
            #print features.shape,"     ",featuresarray.shape

            if featuresarray.shape[0] == 0:
                featuresarray = features
            else:
                featuresarray = np.hstack((featuresarray,features))
                resultarray.append(result)
            #print featuresarray.T , featuresarray.T.shape
            print featuresarray.shape

        min_max_scalar = pr.MinMaxScaler()
        self.features_train = min_max_scalar.fit_transform(featuresarray.T)


        """features = features.T.tolist()
            for i in xrange(0,features.__len__()):
                for each in xrange(0,features[i].__len__()):
                    if self.maxlist[each] < features[i][each]:
                        self.maxlist[each]=features[i][each]
                    elif self.minlist[each] > features[i][each]:
                        self.minlist[each]=features[i][each]

        for x in xrange(0,self.maxlist.__len__()):
            self.denom.append(self.maxlist[x]-self.minlist[x])"""

        print "normalisation variable completed"
        return self.features_train

    def test_emo_db_complete(self,emotion='N'):

        fileaddress = "/home/project/Documents/Project/training/wav/*.wav"
        fileList = glob.glob(fileaddress)

        for fileindex in xrange(0,fileList.__len__()):
            trainWav = fileList[fileindex]
        #for trainWav in fileList:
        #   l,x = wavfile.read(trainWav)

            """segmentsArray,probOnSet = VoiceInput.audioSVMSegmentation(xraw,0.02,0.02,plot = False)

            x = np.array([])
            for segments in segmentsArray:
                if x.shape<=0:
                    x = xraw[segments[0]*self.Fs:segments[1]*self.Fs]
                    continue
                else:
                    x = np.hstack((x,xraw[segments[0]*self.Fs:segments[1]*self.Fs]))"""

        #    features = self.voicefeature.FeatureExtraction(x,x.shape[0],x.shape[0])
            result = trainWav[-6]
            print result

            """if result == 'F':
                y = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif result =='W':
                y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif result =='A':
                y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            elif result =='L':
                y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            elif result =='E':
                y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif result =='T':
                y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif result =='N':
                y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            """

            """features = features.T.tolist()
            for axis0 in xrange(0,features.__len__()):
                normalizedFeatures = []
                for axis1 in xrange(0,features[axis0].__len__()):
                    normalizedFeatures.append((features[axis0][axis1]-self.minlist[axis1])/self.denom[axis1])"""
            response = []
            emo = []
            for res,neural in self.neuralnetwork.items():
                emotioncalculated = neural.forward(self.features_train[fileindex])
                emo.append(emotioncalculated[0])
                response.append([emotioncalculated,result,res])

                #response.index(max(response))
            self.emolist.append(emo)
            print response
        emofile = open("emotion.normal","wb")
        pickle.dump(self.emolist,emofile)
        emofile.close()
        #print self.success, self.failure
        return self.neuralnetwork

if __name__ == "__main__":
    trainingdata = Training()
