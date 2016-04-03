import numpy as np
from voiceInput import VoiceInput
from scipy.fftpack import fft
from scipy.fftpack import dct


class VoiceFeature():
    def __init__(self):
        self.Fs = 16000
        self.eps = 0.00000001

    def ZCR(self, frame):
        count = len(frame)
        countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        return (np.float64(countZ) / np.float64(count-1.0))

    def Energy(self,frame):
        return np.sum(frame ** 2) / np.float64(len(frame))

    def EnergyEntropy(self, frame, numOfShortBlocks=10):
        Eol = np.sum(frame ** 2)    # total frame energy
        L = len(frame)
        subWinLength = int(np.floor(L / numOfShortBlocks))
        if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
        # subWindows is of size [numOfShortBlocks*L]
        subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

        # Compute normalized sub-frame energies:
        s = np.sum(subWindows ** 2, axis=0) / (Eol + self.eps)

        # Compute entropy of the normalized sub-frame energies:
        Entropy = -np.sum(s * np.log2(s + self.eps))
        return Entropy

    def SpectralCentroidAndSpread(self, X, fs):
        ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

        Xt = X.copy()
        Xt = Xt / Xt.max()
        NUM = np.sum(ind * Xt)
        DEN = np.sum(Xt) + self.eps

        # Centroid:
        C = (NUM / DEN)

        # Spread:
        S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

        # Normalize:
        C = C / (fs / 2.0)
        S = S / (fs / 2.0)

        return (C, S)

    def SpectralEntropy(self, X, numOfShortBlocks=10):

        L = len(X)                         # number of frame samples
        Eol = np.sum(X ** 2)            # total spectral energy

        subWinLength = int(np.floor(L / numOfShortBlocks))   # length of sub-frame
        if L != subWinLength * numOfShortBlocks:
            X = X[0:subWinLength * numOfShortBlocks]

        subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames
        s = np.sum(subWindows ** 2, axis=0) / (Eol + self.eps)                      # compute spectral sub-energies
        En = -np.sum(s*np.log2(s + self.eps))                                    # compute spectral entropy

        return En

    def SpectralFlux(self, X, Xprev):
        # compute the spectral flux as the sum of square
        sumX = np.sum(X + self.eps)
        sumPrevX = np.sum(Xprev + self.eps)
        F = np.sum((X / sumX - Xprev/sumPrevX) ** 2)

        return F

    def SpectralRollOff(self, X, c, fs):
        totalEnergy = np.sum(X ** 2)
        fftLength = len(X)
        Thres = c*totalEnergy
        CumSum = np.cumsum(X ** 2) + self.eps
        [a, ] = np.nonzero(CumSum > Thres)
        if len(a) > 0:
            sRO = np.float64(a[0]) / (float(fftLength))
        else:
            sRO = 0.0
        return (sRO)

    def Harmonic(self,frame):
        M = np.round(0.016 * self.Fs) - 1
        R = np.correlate(frame, frame, mode='full')

        g = R[len(frame)-1]
        R = R[len(frame):-1]


        [a, ] = np.nonzero(np.diff(np.sign(R)))

        if len(a) == 0:
            m0 = len(R)-1
        else:
            m0 = a[0]
        if M > len(R):
            M = len(R) - 1

        Gamma = np.zeros((M), dtype=np.float64)
        CSum = np.cumsum(frame ** 2)
        Gamma[m0:M] = R[m0:M] / (np.sqrt((g * CSum[M:m0:-1])) + self.eps)

        ZCR = self.ZCR(Gamma)

        if ZCR > 0.15:
            HR = 0.0
            f0 = 0.0
        else:
            if len(Gamma) == 0:
                HR = 1.0
                blag = 0.0
                Gamma = np.zeros((M), dtype=np.float64)
            else:
                HR = np.max(Gamma)
                blag = np.argmax(Gamma)

        # Get fundamental frequency:
            f0 = self.Fs / (blag + self.eps)
            if f0 > 5000:
                f0 = 0.0
            if HR < 0.1:
                f0 = 0.0

        return (HR, f0)

    def ChromaFeaturesInit(self,nfft, fs):
        freqs = np.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])
        Cp = 27.50

        nChroma = np.round(12.0 * np.log2(freqs+self.eps / Cp)).astype(int)


        nFreqsPerChroma = np.zeros((nChroma.shape[0], ))

        uChroma = np.unique(nChroma)

        for u in uChroma:
            idx = np.nonzero(nChroma == u)
            nFreqsPerChroma[idx] = idx[0].shape
        return nChroma, nFreqsPerChroma

    def ChromaFeatures(self, X, fs, nChroma, nFreqsPerChroma):

        chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        spec = X**2
        C = np.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
        finalC = np.zeros((12, 1))
        newD = int(np.ceil(C.shape[0] / 12.0) * 12)
        C2 = np.zeros((newD, ))
        C2[0:C.shape[0]] = C
        C2 = C2.reshape(C2.shape[0]/12, 12)
        finalC = np.matrix(np.sum(C2, axis=0)).T
        finalC /= spec.sum()

        return chromaNames, finalC

    def mfccInitFilterBanks(self, fs, nfft):
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
        freqs = np.zeros(nFiltTotal+2)
        freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linsc
        freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** np.arange(1, numLogFilt + 3)
        heights = 2./(freqs[2:] - freqs[0:-2])

        # Compute filterbank coeff
        fbank = np.zeros((nFiltTotal, nfft))
        nfreqs = np.arange(nfft) / (1. * nfft) * fs

        for i in range(nFiltTotal):
            lowTrFreq = freqs[i]
            cenTrFreq = freqs[i+1]
            highTrFreq = freqs[i+2]

            lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, np.floor(cenTrFreq * nfft / fs) + 1, dtype=np.int)
            lslope = heights[i] / (cenTrFreq - lowTrFreq)
            rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, np.floor(highTrFreq * nfft / fs) + 1, dtype=np.int)
            rslope = heights[i] / (highTrFreq - cenTrFreq)
            fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
            fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

        return fbank, freqs

    def MFCC(self, X, fbank, nceps):

        mspec = np.log10(np.dot(X, fbank.T) + self.eps)
        ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
        return ceps

    def FeatureExtraction(self, signal, Win, Step):
        Win = int(Win)
        Step = int(Step)

        # Signal normalization
        signal = np.double(signal)
        signal = signal / (2.0 ** 15)
        DC = signal.mean()
        MAX = (np.abs(signal)).max()
        signal = (signal - DC) / MAX

        N = len(signal)                                # total number of samples
        curPos = 0
        countFrames = 0
        nFFT = Win / 2

        [fbank, freqs] = self.mfccInitFilterBanks(self.Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
        nChroma, nFreqsPerChroma = self.ChromaFeaturesInit(nFFT, self.Fs)

        numOfTimeSpectralFeatures = 0
        numOfHarmonicFeatures = 0
        numOfChromaFeatures = 0
        nceps = 13  #MFCC features
        #TODO IGR of harmonic features
        totalNumOfFeatures = numOfTimeSpectralFeatures + nceps +numOfHarmonicFeatures #+numOfChromaFeatures
        Features = np.array([], dtype=np.float64)

        while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
            countFrames += 1
            x = signal[curPos:curPos+Win]                    # get current window
            curPos = curPos + Step                           # update window position
            X = abs(fft(x))                                  # get fft magnitude
            X = X[0:nFFT]                                    # normalize fft
            X = X / len(X)
            if countFrames == 1:
                Xprev = X.copy()                             # keep previous fft mag

            curFV = np.zeros((totalNumOfFeatures, 1))
            #curFV[0] = self.ZCR(x)                              # zero crossing rate
            #curFV[1] = self.Energy(x)                           # short-term energy
            #curFV[2] = self.EnergyEntropy(x)                    # short-term entropy of energy
            #[curFV[3], curFV[4]] = self.SpectralCentroidAndSpread(X, self.Fs)    # spectral centroid and spread
            #curFV[5] = self.SpectralEntropy(X)                  # spectral entropy
            #curFV[6] = self.SpectralFlux(X, Xprev)              # spectral flux
            #curFV[5] = self.SpectralRollOff(X, 0.90, self.Fs)        # spectral rolloff
            #curFV[6] ,curFV [7] = self.Harmonic(x)
            curFV[numOfTimeSpectralFeatures+numOfHarmonicFeatures:numOfTimeSpectralFeatures+numOfHarmonicFeatures+nceps, 0] = self.MFCC(X, fbank, nceps).copy()    # MFCCs
            #curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = MFCC(X, nwin = 2048, nfft = 2048)[0].T.copy()
            #chromaNames, chromaF = self.ChromaFeatures(X, self.Fs, nChroma, nFreqsPerChroma)
            #curFV[numOfTimeSpectralFeatures + nceps +numOfHarmonicFeatures: numOfTimeSpectralFeatures + numOfHarmonicFeatures +nceps + numOfChromaFeatures - 1] = chromaF
            #curFV[numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures -1] = chromaF.std()
            if countFrames == 1:
                Features = curFV                                        # initialize feature matrix (if fir frame)
            else:
                Features = np.concatenate((Features, curFV), 1)    # update feature matrix
            Xprev = X.copy()

        return np.array(Features)

if __name__ == "__main__":
    voiceinput = VoiceInput()
    voicesignal = voiceinput.voiceInput(4)
    voicefeature = VoiceFeature()
    print voicefeature.FeatureExtraction(voicesignal,voicesignal.shape[0],voicesignal.shape[0])