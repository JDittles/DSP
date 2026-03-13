import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft
import math
import matplotlib.pyplot as plt

M = 63
window = get_window('blackmanharris', M)
hM1 = int(math.floor((M + 1) / 2))
hM2 = int(math.floor(M/2))

N = 512
hN = N/2
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = window[hM2:]
fftbuffer[N-hM2:] = window[:hM1-1]

X = fft(fftbuffer)
absX = np.abs(X)
# for any values where the magnitude is less than the smallest
# positive float, set it to the smallest positive float to 
# avoid log(0) issues
absX[absX<np.finfo(float).eps] = np.finfo(float).eps
mX = 20 * np.log10(absX)
pX = np.angle(X)

mX1 = np.zeros(N)
pX1 = np.zeros(N)
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]
pX1[:hN] = pX[hN:]
pX1[N-hN:] = pX[:hN]

plt.plot(np.arange(-hN, hN)/float(N)*M, mX1-max(mX1))
plt.axis([-20, 20, -120, 0])
plt.show()