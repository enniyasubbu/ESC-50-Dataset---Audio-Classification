#!/usr/bin/env python
#Author=@arunodhayan
print "HANDLING IMPORTS..."


import matplotlib.animation as animation
import json
import time
import operator
import argparse
import Tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import time
from scipy.io import wavfile
from pydub import AudioSegment
import pyaudio
import wave
import time
from multiprocessing import Process
import theano
import glob
from lasagne import random as lasagne_random
from lasagne import layers as l
from PyQt4 import QtGui,QtCore
from matplotlib.figure import Figure
import sys
import ui_main
import numpy as np
import pyqtgraph
import SWHear
import AAL_spec as spectrogram
print "...DONE!"

######################## CONFIG #########################
#Fixed random seed
RANDOM_SEED = 1337
RANDOM = np.random.RandomState(RANDOM_SEED)
lasagne_random.set_rng(RANDOM)

#Pre-trained model params
MODEL_PATH = 'mode/'
TRAINED_MODEL = 'AED_AAL_Example_Run_model_epoch_55.pkl'
TEST_DIR="tes/"



#############################################################
def record_audio(AUDIO_FILE):
    #Create audio stream    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # begin recording
    print"* recording audio clip: ",AUDIO_FILE

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    #print"* done recording audio clip:", AUDIO_FILE

    #cleanup objects
    stream.stop_stream()
    stream.close()

    #save frames to audio clips
    print"* sending data to audio file:", AUDIO_FILE
    wf = wave.open(AUDIO_FILE , 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
################### ARGUMENT PARSER #####################
def parse_args():
    
    parser = argparse.ArgumentParser(description='Ambient Assistance Living')
    for x in range(5): 
     x=0
     INPUT="chunk{0}.wav".format(x)
     x +=1
  
    parser.add_argument('--modelname', dest='modelname', help='name of pre-trained model', type=str, default=None)
    parser.add_argument('--speclength', dest='spec_length', help='spectrogram length in seconds', type=int, default=3)
    parser.add_argument('--overlap', dest='spec_overlap', help='spectrogram overlap in seconds', type=int, default=2)
    parser.add_argument('--results', dest='num_results', help='number of results', type=int, default=5)
    parser.add_argument('--confidence', dest='min_confidence', help='confidence threshold', type=float, default=0.01)

    args = parser.parse_args()    

    #single test file or list of files?
    #if isinstance(args.filenames, basestring):
    #    args.filenames = [args.filenames]

    return args

####################  MODEL LOAD  ########################
def loadModel(filename):
    print "IMPORTING MODEL...",
    net_filename = MODEL_PATH + filename

    with open(net_filename, 'rb') as f:
        data = pickle.load(f)

    #for evaluation, we want to load the complete model architecture and trained classes
    net = data['net']
    classes = data['classes']
    im_size = data['im_size']
    im_dim = data['im_dim']
    
    print "DONE!"

    return net, classes, im_size, im_dim

################# PREDICTION FUNCTION ####################
def getPredictionFuntion(net):
    net_output = l.get_output(net, deterministic=True)

    print "COMPILING THEANO TEST FUNCTION...",
    start = time.time()
    test_net = theano.function([l.get_all_layers(NET)[0].input_var], net_output, allow_input_downcast=True)
    print "DONE! (", int(time.time() - start), "s )"

    return test_net

################# PREDICTION POOLING ####################
def predictionPooling(p):
    
    #You can test different prediction pooling strategies here
    #We only use average pooling
    if p.ndim == 2:
        p_pool = np.mean(p, axis=0)
    else:
        p_pool = p

    return p_pool

####################### PREDICT #########################
def predict(img):    

    #transpose image if dim=3
    try:
        img = np.transpose(img, (2, 0, 1))
    except:
        pass

    #reshape image
    img = img.reshape(-1, IM_DIM, IM_SIZE[1], IM_SIZE[0])

    #calling the test function returns the net output
    prediction = TEST_NET(img)[0] 

    return prediction

####################### TESTING #########################
def testFile(path, spec_length, spec_overlap, num_results, confidence_threshold=0.01):

    #time
    start = time.time()
    
    #extract spectrograms from wav-file and process them
    predictions = []
    spec_cnt = 0
    for spec in spectrogram.getMultiSpec(path, seconds=spec_length, overlap=spec_overlap):

        #make prediction
        p = predict(spec)
        spec_cnt += 1

        #stack predictions
        if len(predictions):
            predictions = np.vstack([predictions, p])  
        else:
            predictions = p

    #prediction pooling
    p_pool = predictionPooling(predictions)

    #get class labels for predictions
    p_labels = {}
    for i in range(p_pool.shape[0]):
        if p_pool[i] >= confidence_threshold:
            p_labels[CLASSES[i]] = p_pool[i]

    #sort by confidence and limit results (None returns all results)
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)[:num_results]

    #take time again
    dur = time.time() - start

    return p_sorted, spec_cnt, dur

#################### EXAMPLE USAGE ######################
if __name__ == "__main__":
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1               #stereo
    RATE = 44100
    RECORD_SECONDS = 5         #record chunks of 5 sec
    TOTAL_RECORD_NUMBER = 7000000000   # total chunks to record and play
    AUDIO_DELAY = 5.0          #playback delay in seconds

    x = 0
    
  
    xs = []
    ys = []
 
    while x < TOTAL_RECORD_NUMBER:

        #define audio file clip
        AUDIO_FILE = "audio0.wav".format(x)
        
        #initialize pyaudio
        p = pyaudio.PyAudio()        

        #Kick off record audio function process
        p1 = Process(target = record_audio(AUDIO_FILE))
        p1.start()

        #kick off play audio function process
        #p2 = Process(target = play_audio(AUDIO_FILE))        
        #p2.start()

        p1.join()
        #p2.join()


        #increment record counter
        x += 1
    #adjust config
        args = parse_args()
        
    #load model
        if args.modelname:
         TRAINED_MODEL = args.modelname
        NET, CLASSES, IM_SIZE, IM_DIM = loadModel(TRAINED_MODEL)
        classes=[]
        conf=[]

    #compile test function
        TEST_NET = getPredictionFuntion(NET)
        for files in glob.glob("*.wav"):
         fname,fext=os.path.splitext(files)    
     
         print 'TESTING:', fname
         pred, cnt, dur = testFile(files, args.spec_length, args.spec_overlap, args.num_results, args.min_confidence)    
         print 'TOP PREDICTION(S):'
         for p in pred:
            print '\t', p[0], int(p[1] * 100), '%'
            classes.append(p[0])
            conf.append(int(p[1]*100))
         cam=classes+conf
         plt.ion()
         samplingFrequency, signalData=wavfile.read("audio0.wav")
         fig, axs = plt.subplots(2,1 , figsize=(10, 8))
        
	 axs[0].specgram(signalData,Fs=samplingFrequency)
      
         axs[0].set_xlabel("Time (Sec)")
         axs[0].set_ylabel("Frequency (Hz)")
         axs[0].set_title("Spectrogram")
        
         axs[1].barh(classes,conf)
        
       
         axs[1].set_xlabel("Confidence(%)")
         axs[1].set_ylabel("Classes")
         axs[1].set_title("Real time detection")
     
         fig.tight_layout()
         print 'PREDICTION FOR', cnt, 'SPECS TOOK', int(dur * 1000), 'ms (', int(dur / cnt * 1000) , 'ms/spec', ')', '\n'
        
         plt.show(block=False)
         
         plt.pause(2)
         
         plt.close('all')
      
