import glob
import numpy as np
from scipy.io import wavfile
from voiceFeature import VoiceFeature
from voiceInput import VoiceInput

class Training():
    def __init__(self,neuralnetwork):

        self.maxlist = [-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999]
        self.minlist = [999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999,999]
        self.denom = []
        self.countlist = [0,0,0,0,0,0,0]
        self.Fs = 16000
        self.neuralnetwork = neuralnetwork
        self.voicefeature = VoiceFeature()

    def emo_db_complete_processing(self,emotion='N'):

        fileaddress = "/home/project/Documents/Project/training/wav/"+emotion+"/*.wav"
        fileList = glob.glob(fileaddress)

        for trainWav in fileList:
            l,xraw = wavfile.read(trainWav)
            try:
                segmentsArray,probOnSet = VoiceInput.audioSVMSegmentation(xraw,0.05,0.05,plot = False)
            except:
                continue

            x = np.array([])
            for segments in segmentsArray:
                if x.shape<=0:
                    x = xraw[segments[0]*self.Fs:segments[1]*self.Fs]
                    continue
                else:
                    x = np.hstack((x,xraw[segments[0]*self.Fs:segments[1]*self.Fs]))

            features = VoiceFeature.FeatureExtraction(x,x.shape[0],x.shape[0])
            result = trainWav[-6]

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
                self.countlist[6] = self.countlist[6]+1


            features = features.T.tolist()
            for axis0 in xrange(0,features.__len__()):
                normalizedFeatures = []
                for axis1 in xrange(0,features[axis0].__len__()):
                    normalizedFeatures.append((features[axis0][axis1]-self.minlist[axis1])/self.denom[axis1])

                self.neuralnetwork.emoTrainer.train(normalizedFeatures,y)

        return self.neuralnetwork

    def feature_normalisation(self):
        fileList = glob.glob("/home/project/Documents/Project/training/wav/*/*.wav")
        for trainWav in fileList:
            l,x = wavfile.read(trainWav)

            features = self.voicefeature.FeatureExtraction(x,x.shape[0],x.shape[0])

            features = features.T.tolist()
            for i in xrange(0,features.__len__()):
                for each in xrange(0,features[i].__len__()):
                    if self.maxlist[each] < features[i][each]:
                        self.maxlist[each]=features[i][each]
                    elif self.minlist[each] > features[i][each]:
                        self.minlist[each]=features[i][each]

        for x in xrange(0,self.maxlist.__len__()):
            self.denom.append(self.maxlist[x]-self.minlist[x])




if __name__ == "__main__":
    trainingdata = Training()
