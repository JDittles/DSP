import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models'))
import utilFunctions as UF
import stft as STFT

inputfile = '../sounds/flute-A4.wav'
window = 'hamming'
M = 801
N = 1024
H = 400

(fs, x) = UF.wavread(inputfile)

w = get_window(window, M)

mX, pX = STFT.stftAnal(x, w, N, H)
