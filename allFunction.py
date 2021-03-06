import sys
import os
import shutil
import alsaaudio,audioop
import time
import mlpy
import numpy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.fftpack import dct
from scipy.signal import butter
from scipy.signal import filtfilt
import glob
import speech_recognition as sr
from nn import trainer
from nn import neuralNet
import main
import pickle
import thread
from PyQt4 import QtCore, QtGui



Fs = 16000
eps = 0.00000001
lowcut = 0
highcut =0
neuralNetwork = neuralNet("test.nn")
emoTrainer = trainer(neuralNetwork)
maxlist = [-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9]
minlist = [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]
denom = []


def guiWrite(gui,text):
	gui.ui.featureText.append(text)
	return

def butter_bandpass(lowcut, highcut, Fs, order):
	#nyq = 0.5 * Fs
	#low = lowcut / nyq
	#high = highcut / nyq
	filtb, filta = butter(order, 0.1,btype = 'low')
	return filtb, filta

def recordAudioSegments(RecordPath, Bs, plot=False):	#Bs: BlockSize
	"""RecordPath += os.sep
	d = os.path.dirname(RecordPath)
	if os.path.exists(d) and RecordPath!=".":
		shutil.rmtree(RecordPath)
	os.makedirs(RecordPath)"""

	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
	inp.setchannels(1)
	inp.setrate(Fs)
	inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	inp.setperiodsize(512)
	midTermBufferSize = int(Fs*Bs)
	midTermBuffer = []
	curWindow = []
	elapsedTime = "%08.3f" % (time.time())
	while 1:
		l,data = inp.read()
		#TODO NOISE Cancellation
		#print "Now listening:"
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
			midTermBufferArray = numpy.int16(midTermBuffer)
			#curWavFileName = RecordPath + os.sep + str(elapsedTime) + ".wav"
			#print midTermBuffer.__class__

			#TODO Noise Filtering
			#filtb, filta = butter_bandpass(lowcut,highcut,Fs,8)
			#midTermBufferArray = filtfilt(filtb , filta , midTermBufferArray)
			#wavfile.write(curWavFileName, Fs, midTermBufferArray)
			#TODO FeatureExtraction call to segmentation
			#TODO Add call to segment voice i.e. svmSegmentation function
			#adaptfilt.lms()
			try:
				segmentsArray,ProbOnSet = svmSegmentation(midTermBufferArray,Fs,0.05,0.05,plot = False)
			except:
				continue

			x = numpy.array([])
			for segments in segmentsArray:
				if x.shape<=0:
					x = midTermBufferArray[segments[0]*Fs:segments[1]*Fs]
					continue
				else:
					x = numpy.hstack((x,midTermBufferArray[segments[0]*Fs:segments[1]*Fs]))

			if plot == 'seg':
				plotSegments(midTermBufferArray,Fs,segmentsArray,ProbOnSet)
				break
			Features = FeatureExtraction(midTermBufferArray,Fs,x.shape[0],x.shape[0])
			#guiWrite(gui,"text")
			features = Features.T.tolist()
			for axis0 in xrange(0,features.__len__()):
				normalizedFeatures = []
				for axis1 in xrange(0,features[axis0].__len__()):
					normalizedFeatures.append((features[axis0][axis1]-minlist[axis1])/denom[axis1])
				#print normalizedFeatures
			plotdata = []
			res = neuralNetwork.forward(normalizedFeatures)


			fig, ax =plt.subplots()
			ar = numpy.arange(7)
			print res
			barPlot = ax.bar(ar,res,0.35)
			plt.show()


			print "Happiness:",res[0],"Anger:",res[1],"Fear:",res[2],"Boredom:",res[3],"Disgust:",res[4],"Sadness:",res[5],"Neutral:",res[6]
			if plot == 'energy':
				plotEnergy(Features[1])
			#print "AUDIO  OUTPUT: Saved " + curWavFileName
			midTermBuffer = []
			elapsedTime = "%08.3f" % (time.time())

def googleRecognition(recognizer, audio):
	if gui == None:
		try:
			print(recognizer.recognize_google(audio))
		except sr.UnknownValueError:
			print("Google Speech Recognition could not understand audio")
		except sr.RequestError as e:
			print("Could not request results from Google Speech Recognition service; {0}".format(e))
	else:
		try:
			gui.ui.speechText.append(recognizer.recognize_google(audio))
		except sr.UnknownValueError:
			gui.ui.speechText.append("Google Speech Recognition could not understand audio")
		except sr.RequestError as e:
			gui.ui.speechText.append("Could not request results from Google Speech Recognition service; {0}".format(e))

def speechRecognition(GUI = None):
	global gui
	gui = GUI
	r = sr.Recognizer()
	m = sr.Microphone()
	with m as source:
    		r.adjust_for_ambient_noise(source) 
	r.listen_in_background(m, googleRecognition)

def ZCR(frame):
	count = len(frame)
	countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
	return (numpy.float64(countZ) / numpy.float64(count-1.0))

def Energy(frame):
	return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def EnergyEntropy(frame, numOfShortBlocks=10):
	Eol = numpy.sum(frame ** 2)    # total frame energy
	L = len(frame)
	subWinLength = int(numpy.floor(L / numOfShortBlocks))
	if L != subWinLength * numOfShortBlocks:
		frame = frame[0:subWinLength * numOfShortBlocks]
	# subWindows is of size [numOfShortBlocks*L]
	subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

	# Compute normalized sub-frame energies:
	s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

	# Compute entropy of the normalized sub-frame energies:
	Entropy = -numpy.sum(s * numpy.log2(s + eps))
	return Entropy

def SpectralCentroidAndSpread(X, fs):
	ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

	Xt = X.copy()
	Xt = Xt / Xt.max()
	NUM = numpy.sum(ind * Xt)
	DEN = numpy.sum(Xt) + eps

	# Centroid:
	C = (NUM / DEN)

	# Spread:
	S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

	# Normalize:
	C = C / (fs / 2.0)
	S = S / (fs / 2.0)

	return (C, S)

def SpectralEntropy(X, numOfShortBlocks=10):

	L = len(X)                         # number of frame samples
	Eol = numpy.sum(X ** 2)            # total spectral energy

	subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
	if L != subWinLength * numOfShortBlocks:
		X = X[0:subWinLength * numOfShortBlocks]

	subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames
	s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
	En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

	return En

#TODO IGR for flux
def SpectralFlux(X, Xprev):
	# compute the spectral flux as the sum of square
	sumX = numpy.sum(X + eps)
	sumPrevX = numpy.sum(Xprev + eps)
	F = numpy.sum((X / sumX - Xprev/sumPrevX) ** 2)

	return F

#TODO IGR for rolloff
def SpectralRollOff(X, c, fs):
	totalEnergy = numpy.sum(X ** 2)
	fftLength = len(X)
	Thres = c*totalEnergy
	CumSum = numpy.cumsum(X ** 2) + eps
	[a, ] = numpy.nonzero(CumSum > Thres)
	if len(a) > 0:
		sRO = numpy.float64(a[0]) / (float(fftLength))
	else:
		sRO = 0.0
	return (sRO)

def mfccInitFilterBanks(fs, nfft):

	# filter bank params:
	lowfreq = 133.33
	linsc = 200/3.
	logsc = 1.0711703
	numLinFiltTotal = 13
	numLogFilt = 27

	if fs < 8000:
		nlogfil = 5

		# Total number of filters
	nFiltTotal = numLinFiltTotal + numLogFilt

	# Compute frequency points of the triangle:
	freqs = numpy.zeros(nFiltTotal+2)
	freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
	freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
	heights = 2./(freqs[2:] - freqs[0:-2])

	# Compute filterbank coeff
	fbank = numpy.zeros((nFiltTotal, nfft))
	nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

	for i in range(nFiltTotal):
		lowTrFreq = freqs[i]
		cenTrFreq = freqs[i+1]
		highTrFreq = freqs[i+2]

		lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, numpy.floor(cenTrFreq * nfft / fs) + 1, dtype=numpy.int)
		lslope = heights[i] / (cenTrFreq - lowTrFreq)
		rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, numpy.floor(highTrFreq * nfft / fs) + 1, dtype=numpy.int)
		rslope = heights[i] / (highTrFreq - cenTrFreq)
		fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
		fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

	return fbank, freqs

def MFCC(X, fbank, nceps):

	mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
	ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
	return ceps

def FeatureExtraction(signal, Fs, Win, Step):
	Win = int(Win)
	Step = int(Step)

	# Signal normalization
	signal = numpy.double(signal)
	signal = signal / (2.0 ** 15)
	DC = signal.mean()
	MAX = (numpy.abs(signal)).max()
	signal = (signal - DC) / MAX

	N = len(signal)                                # total number of samples
	curPos = 0
	countFrames = 0
	nFFT = Win / 2

	[fbank, freqs] = mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation

	numOfTimeSpectralFeatures = 6
	#numOfHarmonicFeatures = 0
	nceps = 13  #MFCC features
	#TODO IGR of harmonic features
	totalNumOfFeatures = numOfTimeSpectralFeatures + nceps #+numOfHarmonicFeatures
	Features = numpy.array([], dtype=numpy.float64)

	while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
		countFrames += 1
		x = signal[curPos:curPos+Win]                    # get current window
		curPos = curPos + Step                           # update window position
		X = abs(fft(x))                                  # get fft magnitude
		X = X[0:nFFT]                                    # normalize fft
		X = X / len(X)
		if countFrames == 1:
			Xprev = X.copy()                             # keep previous fft mag
		#TODO WARNING!!! DO NOT ALTER THE SEQUENCE OF THESE FEATURES
		#TODO THESE ARE USED ONLY BASED ON SEQUENCE FOR SEGMENTATION
		#TODO curFV is most important variable DO NOT PLAY WITH IT
		curFV = numpy.zeros((totalNumOfFeatures, 1))
		curFV[0] = ZCR(x)                              # zero crossing rate
		curFV[1] = Energy(x)                           # short-term energy
		curFV[2] = EnergyEntropy(x)                    # short-term entropy of energy
		[curFV[3], curFV[4]] = SpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
		#curFV[5] = SpectralEntropy(X)                  # spectral entropy
		#curFV[6] = SpectralFlux(X, Xprev)              # spectral flux
		curFV[5] = SpectralRollOff(X, 0.90, Fs)        # spectral rolloff
		curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = MFCC(X, fbank, nceps).copy()    # MFCCs
		#curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = MFCC(X, nwin = 2048, nfft = 2048)[0].T.copy()
		if countFrames == 1:
			Features = curFV                                        # initialize feature matrix (if fir frame)
		else:
			Features = numpy.concatenate((Features, curFV), 1)    # update feature matrix
		Xprev = X.copy()
	
	#print Features.shape
	#raw_input('features extracted:')
	return numpy.array(Features)

def smoothMovingAvg(inputSignal,windowLen):

	if inputSignal.ndim != 1:
		raise ValueError("")
	if inputSignal.size < windowLen:
		raise ValueError("Input vector needs to be bigger than window size.")
	if windowLen < 3:
		return inputSignal
	s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1], inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
	w = numpy.ones(windowLen, 'd')
	y = numpy.convolve(w/w.sum(), s, mode='same')
	return y[windowLen:-windowLen+1]

def stereo2mono(x):
	if x.ndim==1:
		return x
	else:
		if x.ndim==2:
			return ( (x[:,1] / 2) + (x[:,0] / 2) )
		else:
			return -1

def normalizeFeatures(features):
	X = numpy.array([],dtype=float)

	for count, f in enumerate(features):
		if f.shape[0] > 0:
			if count == 0:
				X = f
			else:
				if X.__len__() == 0 :
					X=f
				#print "Vstack dimension",X.shape,f.shape
				else:
					X = numpy.vstack((X, f))
		count += 1

	MEAN = numpy.mean(X, axis=0)
	STD = numpy.std(X, axis=0)

	featuresNorm = []
	for f in features:
		ft = f.copy()
		for nSamples in range(f.shape[0]):
			ft[nSamples, :] = (ft[nSamples, :] - MEAN) / STD
		featuresNorm.append(ft)
	return (featuresNorm, MEAN, STD)

def trainSVM(features, Cparam):
	#TODO Call mlpy to train SVM
	X = numpy.array([])
	Y = numpy.array([])
	for i, f in enumerate(features):
		if i == 0:
			X = f
			Y = i * numpy.ones((len(f), 1))
		else:
			X = numpy.vstack((X, f))
			Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
	svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='linear', eps=0.0000001, C=Cparam, probability=True)
	svm.learn(X, Y)
	return svm

def energyExtraction(x,Fs,Win, Step):
	Win = int(Win)
	Step = int(Step)

	# Signal normalization
	signal = numpy.double(x)
	signal = signal / (2.0 ** 15)
	DC = signal.mean()
	MAX = (numpy.abs(signal)).max()
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
		curFV = numpy.zeros((2, 1))
		curFV[0] = countFrames
		curFV[1] = Energy(x)                           # short-term energy
		if countFrames == 1:
			energyFeatures = curFV                                        # initialize feature matrix (if fir frame)
		else:
			energyFeatures = numpy.concatenate((energyFeatures, curFV), 1)    # update feature matrix
		Xprev = X.copy()


	return energyFeatures
	
def svmSegmentation(x, Fs, window, steps, plot=True):
	smoothWindow=1
	Weight=0.3
	#TODO Step 1: energy extraction
	#TODO stereo2mono function defination to be rendered 
	#TODO Warning stereo format is not relaible cause of multiple channels
	x = stereo2mono(x)                        # convert to mono
	ShortTermFeatures = energyExtraction(x, Fs, window * Fs, steps * Fs)        # extract short-term features
	#TODO ShortTermFeatures usage checking
	#TODO Step 2: train binary SVM classifier of low vs high energy frames
	EnergySt = ShortTermFeatures[1, :]                  # keep only the energy short-term sequence 
	#print ShortTermFeatures
	E = numpy.sort(EnergySt)                            # sort the energy feature values:
	L1 = int(len(E) / 10)                               # number of 10% of the total short-term windows
	T1 = numpy.mean(E[0:L1])                           # compute "lower" 10% energy threshold

	T2 = numpy.mean(E[-L1:-1])                          # compute "higher" 10% energy threshold
	Class1 = ShortTermFeatures[:, numpy.where(EnergySt < T1)[0]]         # get all features that correspond to low energy
	#print "Class1:",Class1.shape,Class1
	Class2 = ShortTermFeatures[:, numpy.where(EnergySt > T2)[0]]         # get all features that correspond to high energy
	#print "Class2:",Class2.shape,Class2
	featuresSS = [Class1.T, Class2.T]                                   # form features for svm segmentation
	#print featuresSS	
	[featuresNormSS, MEANSS, STDSS] = normalizeFeatures(featuresSS)   # normalize according to mag for segmentation
	#TODO add featuresNormSS
	SVM = trainSVM(featuresNormSS, 1.0)                               # train the respective SVM probabilistic model (ONSET vs SILENCE)

	#TODO Step 3: compute onset probability based on the trained SVM
	ProbOnset = []
	for i in range(ShortTermFeatures.shape[1]):                    # for each frame
		curFV = (ShortTermFeatures[:, i] - MEANSS) / STDSS         # normalize feature vector
		#print curFV,ShortTermFeatures[:,i],MEANSS
		ProbOnset.append(SVM.pred_probability(curFV)[1])           # get SVM probability (that it belongs to the ONSET class)
	ProbOnset = numpy.array(ProbOnset)
	ProbOnset = smoothMovingAvg(ProbOnset, smoothWindow / steps)  # smooth probability

	#TODO Step 4: detect onset frame indices:
	ProbOnsetSorted = numpy.sort(ProbOnset)                        # find probability Threshold as a weighted average of top 10% and lower 10% of the values
	Nt = ProbOnsetSorted.shape[0] / 10
	T = (numpy.mean((1 - Weight) * ProbOnsetSorted[0:Nt]) + Weight * numpy.mean(ProbOnsetSorted[-Nt::]))
	#print "T:",T
	MaxIdx = numpy.where(ProbOnset > T)[0]                         # get the indices of the frames that satisfy the thresholding
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

	#TODO Step 6: Post process: remove very small segments:
	"""minDuration = 0.2
	segmentLimits2 = []
	for s in segmentLimits:
		if s[1] - s[0] > minDuration:
			segmentLimits2.append(s)
	segmentLimits = segmentLimits2"""
	if plot==True:
		plotSegments(x, Fs, segmentLimits, ProbOnset)
	#print "Segmentation;",segmentLimits
	return segmentLimits,ProbOnset

def plotSegments(x, Fs, segmentLimits, ProbOnset):
	steps = 0.02
	timeX = numpy.arange(0, x.shape[0] / float(Fs), 1.0 / Fs)

	plt.subplot(2, 1, 1)
	plt.plot(timeX, x)
	for s in segmentLimits:
		plt.axvline(x=s[0])
		plt.axvline(x=s[1])
	plt.subplot(2, 1, 2)
	plt.plot(numpy.arange(0, ProbOnset.shape[0] * steps, steps), ProbOnset)
	plt.title('Signal')
	for s in segmentLimits:
		plt.axvline(x=s[0])
		plt.axvline(x=s[1])
	plt.title('SVM Probability')
	plt.show()
	return

def plotEnergy():
	return

def train():
	fileList = glob.glob("/home/project/Documents/Project/training/wav/*.wav")
	for trainWav in fileList:
		l,x = wavfile.read(trainWav)
		"""try:
			segmentsArray,probOnSet = svmSegmentation(xraw,Fs,0.05,0.05,plot = False)
		#print segments
		except:
			continue

		x = numpy.array([])
		for segments in segmentsArray:
			if x.shape<=0:
				x = xraw[segments[0]*Fs:segments[1]*Fs]
				continue
			else:
				#print x.shape,xraw.shape
				x = numpy.hstack((x,xraw[segments[0]*Fs:segments[1]*Fs]))"""

		features = FeatureExtraction(x,Fs,x.shape[0],x.shape[0])

		features = features.T.tolist()
		for i in xrange(0,features.__len__()):
			for each in xrange(0,features[i].__len__()):
				if maxlist[each] < features[i][each]:
					maxlist[each]=features[i][each]
				elif minlist[each] > features[i][each]:
					minlist[each]=features[i][each]

	for x in xrange(0,maxlist.__len__()):
		denom.append(maxlist[x]-minlist[x])

	countlist = [0,0,0,0,0,0,0]

	for trainWav in fileList:
		l,xraw = wavfile.read(trainWav)
		try:
			segmentsArray,probOnSet = svmSegmentation(xraw,Fs,0.05,0.05,plot = False)
		except:
			continue

		x = numpy.array([])
		for segments in segmentsArray:
			if x.shape<=0:
				x = xraw[segments[0]*Fs:segments[1]*Fs]
				continue
			else:
				x = numpy.hstack((x,xraw[segments[0]*Fs:segments[1]*Fs]))

		features = FeatureExtraction(x,Fs,x.shape[0],x.shape[0])
		result = trainWav[-6]

		if result == 'F':
			y = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

			if countlist[0] >= 58:
				continue
			countlist[0] = countlist[0]+1
		elif result =='W':
			y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

			if countlist[1] >= 70:
				continue
			countlist[1] = countlist[1]+1
		elif result =='A':
			y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

			if countlist[2] >= 60:
				continue
			countlist[2] = countlist[2]+1
		elif result =='L':
			y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

			if countlist[3] >= 50:
				continue
			countlist[3] = countlist[3]+1

		elif result =='E':
			y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

			if countlist[4] >= 60:
				continue
			countlist[4] = countlist[4]+1

		elif result =='T':
			y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

			if countlist[5] >= 60:
				continue
			countlist[5] = countlist[5]+1
		elif result =='N':
			y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

			if countlist[6] >= 60:
				continue
			countlist[6] = countlist[6]+1


		features = features.T.tolist()
		for axis0 in xrange(0,features.__len__()):
			normalizedFeatures = []
			for axis1 in xrange(0,features[axis0].__len__()):
				normalizedFeatures.append((features[axis0][axis1]-minlist[axis1])/denom[axis1])
			#print normalizedFeatures

			emoTrainer.train(normalizedFeatures,y)


	for trainWav in fileList:
		l,xraw = wavfile.read(trainWav)
		try:
			segmentsArray,probOnSet = svmSegmentation(xraw,Fs,0.05,0.05,plot = False)
		except:
			continue

		x = numpy.array([])
		for segments in segmentsArray:
			if x.shape<=0:
				x = xraw[segments[0]*Fs:segments[1]*Fs]
				continue
			else:
				x = numpy.hstack((x,xraw[segments[0]*Fs:segments[1]*Fs]))

		features = FeatureExtraction(x,Fs,x.shape[0],x.shape[0])
		print trainWav[-6]

		print "[F,W,A,L,E,T,N]"
		features = features.T.tolist()
		finalResponse = []
		for axis0 in xrange(0,features.__len__()):
			normalizedFeatures = []
			for axis1 in xrange(0,features[axis0].__len__()):
				normalizedFeatures.append((features[axis0][axis1]-minlist[axis1])/denom[axis1])
			finalResponse.append( neuralNetwork.forward(normalizedFeatures))
		print finalResponse
		res =  sum(numpy.array(finalResponse)/features.__len__())
	print countlist
	total = sum(res)
	factor = res / total

	max(factor)

	
	fileBuffer  = open(filename,"w")
	pickle.dump(neuralNetwork,fileBuffer)

if __name__ == "__main__":
	speechRecognition()
	while 1:
		print("Select your options:\n1.Train and Test \n2.Input and Analyze\n3.Voice analysis\nInput:")
		optionInput = int(raw_input())
		if optionInput == 1:
			train()
		if optionInput == 2:
			main.control()
		if optionInput == 3:
			recordAudioSegments("",4)
