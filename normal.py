import pickle
import numpy as np
bufferdata = open("emotion.normal","rb")
emotionarray = pickle.load(bufferdata)
bufferdata.close()

emotionarray =np.array(emotionarray)
sumarray = np.sum(emotionarray,axis=0)
avgarray = sumarray/341

subavgarray = np.subtract(emotionarray,avgarray)

normfile = open("voice.normal","wb")
pickle.dump(avgarray,normfile)
normfile.close()

mul100array = subavgarray*100
#print mul100array,subavgarray.sum()
percarray = np.divide(subavgarray.T,np.sum(subavgarray,axis=1)).T

print avgarray,subavgarray,percarray[7],percarray[15]