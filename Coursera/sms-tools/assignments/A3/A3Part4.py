import math
import sys
sys.path.append('../../software/models/')
from dftModel import dftAnal, dftSynth
# We'll use Essentia instead
# import essentia.standard as es
from scipy.signal import get_window
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import utilFunctions as UF
"""
A3-Part-4: Suppressing frequency components using DFT model

Given a frame of the signal, write a function that uses the dftModel functions to suppress all the 
frequency components <= 70Hz in the signal and returns the output of the dftModel 
with and without filtering. 

You will use the DFT model to implement a very basic form of filtering to suppress frequency components. 
When working close to mains power lines, there is a 50/60 Hz hum that can get introduced into the 
audio signal. You will try to remove that using a basic DFT model based filter. You will work on just 
one frame of a synthetic audio signal to see the effect of filtering. 

You can use the functions dftAnal and dftSynth provided by the dftModel file of sms-tools. Use dftAnal 
to obtain the magnitude spectrum (in dB) and phase spectrum of the audio signal. Set the values of 
the magnitude spectrum that correspond to frequencies <= 70 Hz to -120dB (there may not be a bin 
corresponding exactly to 70 Hz, choose the nearest bin of equal or higher frequency, e.g., using np.ceil()).
If you have doubts converting from frequency (Hz) to bins, you can review the beginning of theory lecture 2T1.

Use dftSynth to synthesize the filtered output signal and return the output. The function should also return the 
output of dftSynth without any filtering (without altering the magnitude spectrum in any way). 
You will use a hamming window to smooth the signal. Hence, do not forget to scale the output signals 
by the sum of the window values (as done in sms-tools/software/models_interface/dftModel_function.py). 
To understand the effect of filtering, you can plot both the filtered output and non-filtered output 
of the dftModel. 

Please note that this question is just for illustrative purposes and filtering is not usually done 
this way - such sharp cutoffs introduce artifacts in the output. 

The input is a M length input signal x that contains undesired frequencies below 70 Hz, sampling 
frequency fs and the FFT size N. The output is a tuple with two elements (y, yfilt), where y is the 
output of dftModel with the unaltered original signal and yfilt is the filtered output of the dftModel.

Caveat: In python (as well as numpy) variable assignment is by reference. if you assign B = A, and 
modify B, the value of A also gets modified. If you do not want this to happen, consider using B = A.copy(). 
This creates a copy of A and assigns it to B, and hence, you can modify B without affecting A.

Test case 1: For an input signal with 40 Hz, 100 Hz, 200 Hz, 1000 Hz components, yfilt will only contain
100 Hz, 200 Hz and 1000 Hz components. 

Test case 2: For an input signal with 23 Hz, 36 Hz, 230 Hz, 900 Hz, 2300 Hz components, yfilt will only contain
230 Hz, 900 Hz and 2300 Hz components. 
"""
def suppressFreqDFTmodel(x, fs, N):
    """
    Inputs:
        x (numpy array) = input signal of length M (odd)
        fs (float) = sampling frequency (Hz)
        N (positive integer) = FFT size
    Outputs:
        The function should return a tuple (y, yfilt)
        y (numpy array) = Output of the dftSynth() without filtering (M samples long)
        yfilt (numpy array) = Output of the dftSynth() with filtering (M samples long)
    The first few lines of the code have been written for you, do not modify it. 
    """
    M = len(x)
    seventy_hz_bin = int(math.ceil((70 * N) / float(fs)))
    print("Seventy Hz bin: " + str(seventy_hz_bin))
    hM1 = int(math.floor((M+1)/2))
    hM2 = int(math.floor(M/2))
    fftbuffer = np.zeros(N)
    fftbuffer[:hM1] = x[hM2:]
    fftbuffer[N-hM2:] = x[:hM2]
    X = fft(fftbuffer, N)
    mX = 20 * np.log10(np.abs(X[:(N//2)+1]))
    pX = np.unwrap(np.angle(X[:(N//2)+1]))
    y = dftSynth(mX, pX, N)
    mX[0:seventy_hz_bin] = -120
    y_filt = dftSynth(mX, pX, N)
    # plt.plot(x)
    # plt.show()
    # plt.plot(fftbuffer)
    # plt.show()
    # plt.plot(mX)
    # plt.show()
    # plt.plot(pX)
    # plt.show()
    # plt.plot(y)
    # plt.show()
    # plt.plot(y_filt)
    # plt.show()
    return (y, y_filt)

# x = np.sin(2*np.pi*40*np.arange(0, 140)/16000) + np.sin(2*np.pi*100*np.arange(0, 140)/16000) + np.sin(2*np.pi*200*np.arange(0, 140)/16000) + np.sin(2*np.pi*1000*np.arange(0, 140)/16000)
# fs = 16000  # Example sampling frequency
# N = 4096  # Example FFT size
# print(len(x))
# suppressFreqDFTmodel(x, fs, N)
