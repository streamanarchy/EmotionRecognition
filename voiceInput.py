import alsaaudio
import audioop
import numpy as np
from scipy.fftpack import fft
import mlpy

class VoiceInput():
    def __init__(self):
        self.Fs = 16000

    def voiceInput(self,Bs):
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
        inp.setchannels(1)
        inp.setrate(self.Fs)
        inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        inp.setperiodsize(512)

        midTermBufferSize = int((self.Fs)*Bs)

        midTermBuffer = []
        curWindow = []

        while 1:
            l,data = inp.read()
            #TODO NOISE Cancellation

            if l:
                for i in range(len(data)/2):
                    curWindow.append(audioop.getsample(data, 2, i))

                if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
                    samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
                else:
                    samplesToCopyToMidBuffer = len(curWindow)

                midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer]
                del(curWindow[0:samplesToCopyToMidBuffer])


            if len(midTermBuffer) == midTermBufferSize:
                midTermBufferArray = np.int16(midTermBuffer)

                #TODO Noise Filtering
                #filtb, filta = butter_bandpass(lowcut,highcut,Fs,8)
                #midTermBufferArray = filtfilt(filtb , filta , midTermBufferArray)
                #wavfile.write(curWavFileName, Fs, midTermBufferArray)

                print midTermBufferArray
                segmentsArray,ProbOnSet = self.audioSVMSegmentation(midTermBufferArray,0.02,0.02,plot = False)

                x = np.array([])
                for segments in segmentsArray:
                    if x.shape<=0:
                        x = midTermBufferArray[segments[0]*self.Fs:segments[1]*self.Fs]
                        continue
                    else:
                        x = np.hstack((x,midTermBufferArray[segments[0]*self.Fs:segments[1]*self.Fs]))
                return x
            #midTermBuffer = []

    def stereo2mono(self,x):
        if x.ndim==1:
            return x
        else:
            if x.ndim==2:
                return ( (x[:,1] / 2) + (x[:,0] / 2) )
            else:
                return -1

    def normalizeFeatures(self,features):
        X = np.array([],dtype=float)

        for count, f in enumerate(features):
            if f.shape[0] > 0:
                if count == 0:
                    X = f
                else:
                    if X.__len__() == 0 :
                        X=f
                    else:
                        X = np.vstack((X, f))
            count += 1

        MEAN = np.mean(X, axis=0)
        STD = np.std(X, axis=0)

        featuresNorm = []
        for f in features:
            ft = f.copy()
            for nSamples in range(f.shape[0]):
                ft[nSamples, :] = (ft[nSamples, :] - MEAN) / STD
            featuresNorm.append(ft)
        return (featuresNorm, MEAN, STD)

    def Energy(self,frame):
        return np.sum(frame ** 2) / np.float64(len(frame))

    def energyExtraction(self,x,Win, Step):
        Win = int(Win)
        Step = int(Step)

        # Signal normalization
        signal = np.double(x)
        signal = signal / (2.0 ** 15)
        DC = signal.mean()
        MAX = (np.abs(signal)).max()
        signal = (signal - DC) / MAX

        N = len(signal)                                # total number of samples
        curPos = 0
        countFrames = 0
        nFFT = Win / 2

        while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
            countFrames += 1
            x = signal[curPos:curPos+Win]                    # get current window
            curPos = curPos + Step                           # update window position
            X = abs(fft(x))                                  # get fft magnitude
            X = X[0:nFFT]                                    # normalize fft
            X = X / len(X)
            if countFrames == 1:
                Xprev = X.copy()                             # keep previous fft mag
            curFV = np.zeros((2, 1))
            curFV[0] = countFrames
            print countFrames
            curFV[1] = self.Energy(x)                          # short-term energy
            if countFrames == 1:
                energyFeatures = curFV                                        # initialize feature matrix (if fir frame)
            else:
                energyFeatures = np.concatenate((energyFeatures, curFV), 1)    # update feature matrix
            Xprev = X.copy()


        return energyFeatures

    def trainSVM(self,features, Cparam):
        X = np.array([])
        Y = np.array([])
        for i, f in enumerate(features):
            if i == 0:
                X = f
                Y = i * np.ones((len(f), 1))
            else:
                X = np.vstack((X, f))
                Y = np.append(Y, i * np.ones((len(f), 1)))
        svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='linear', eps=0.0000001, C=Cparam, probability=True)
        svm.learn(X, Y)
        return svm

    def smoothMovingAvg(self,inputSignal,windowLen):

        if inputSignal.ndim != 1:
            raise ValueError("")
        if inputSignal.size < windowLen:
            print windowLen, inputSignal
            raise ValueError("Input vector needs to be bigger than window size.")
        if windowLen < 3:
            return inputSignal
        s = np.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1], inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
        w = np.ones(windowLen, 'd')
        y = np.convolve(w/w.sum(), s, mode='same')
        return y[windowLen:-windowLen+1]

    def audioSVMSegmentation(self,x,window,steps,plot=False):
        smoothWindow=1
        Weight=0.3
        
        x = self.stereo2mono(x)                        # convert to mono

        energy = self.energyExtraction(x, window * self.Fs, steps * self.Fs)        # extract short-term features

        EnergySt = energy[1, :]                  # keep the energy short-term sequence

        E = np.sort(EnergySt)                            # sort the energy feature values:
        L1 = int(len(E) / 10)                               # number of 10% of the total short-term windows
        T1 = np.mean(E[0:L1])                           # compute "lower" 10% energy threshold

        T2 = np.mean(E[-L1:-1])                          # compute "higher" 10% energy threshold
        Class1 = energy[:, np.where(EnergySt < T1)[0]]         # get all features that correspond to low energy
        Class2 = energy[:, np.where(EnergySt > T2)[0]]         # get all features that correspond to high energy
        featuresSS = [Class1.T, Class2.T]                                   # form features for svm segmentation
        [featuresNormSS, MEANSS, STDSS] = self.normalizeFeatures(featuresSS)   # normalize according to mag for segmentation

        SVM = self.trainSVM(featuresNormSS, 1.0)                               # train the respective SVM probabilistic model (ONSET vs SILENCE)

        ProbOnset = []
        for i in range(energy.shape[1]):                    # for each frame
            curFV = (energy[:, i] - MEANSS) / STDSS         # normalize feature vector
            #print curFV,ShortTermFeatures[:,i],MEANSS
            print curFV[1]
            print SVM.pred_probability(curFV)

            ProbOnset.append(SVM.pred_probability(curFV)[1])           # get SVM probability (that it belongs to the ONSET class)
        ProbOnset = np.array(ProbOnset)

        ProbOnset = self.smoothMovingAvg(ProbOnset, smoothWindow / steps)  # smooth probability

        #TODO Step 4: detect onset frame indices:
        ProbOnsetSorted = np.sort(ProbOnset)                        # find probability Threshold as a weighted average of top 10% and lower 10% of the values
        Nt = ProbOnsetSorted.shape[0] / 10
        T = (np.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * np.mean(ProbOnsetSorted[-Nt::]))
        #print "T:",T
        MaxIdx = np.where(ProbOnset > T)[0]                         # get the indices of the frames that satisfy the thresholding
        i = 0
        timeClusters = []
        segmentLimits = []
        #print MaxIdx
        #TODO Step 5: group frame indices to onset segments
        while i < len(MaxIdx):                                         # for each of the detected onset indices
            curCluster = [MaxIdx[i]]
            if i == len(MaxIdx)-1:
                break

            while MaxIdx[i+1] - curCluster[-1] <= 2:
                curCluster.append(MaxIdx[i+1])
                i += 1
                if i == len(MaxIdx)-1:
                    break
            i += 1

            timeClusters.append(curCluster)
            segmentLimits.append([curCluster[0] * steps, curCluster[-1] * steps])

        """minDuration = 0.2
        segmentLimits2 = []
        for s in segmentLimits:
            if s[1] - s[0] > minDuration:
                segmentLimits2.append(s)
        segmentLimits = segmentLimits2"""
        """if plot==True:
            plotSegments(x, self.Fs, segmentLimits, ProbOnset)
        #print "Segmentation;",segmentLimits"""
        return segmentLimits,ProbOnset

if __name__ == "__main__":
    voiceinput = VoiceInput()
    voiceinput.voiceInput(4)
