import allFunction as aF
import thread

def call():
	thread.start_new_thread(aF.recordAudioSegments,('/home/project/Documents/Project/input/wav/',4))
	thread.start_new_thread(aF.speechRecognition,())

	while 1:	
		pass

if __name__ == "__main__":
	call()
