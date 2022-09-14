import wave
import numpy as np
import os

# load data rom wav file
def load_wav_file(filename):
	buf = bytearray(os.path.getsize(filename))
	with open(filename, 'rb') as f:
		f.readinto(buf)
	return buf
if __name__=="__main__":
	data = load_wav_file('zsmeif.wav')
	print(data.shape)