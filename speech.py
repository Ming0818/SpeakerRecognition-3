import sys
import numpy as np
import os                                             # for reading files from disk.
import random as rd                                   # for the Gaussian Montecarlo.

from scipy.io import wavfile                          # for reading sound files.
from sklearn.ensemble import RandomForestClassifier   # for machine learning.
from sklearn.externals import joblib                  # for saving the trained classifier. 

sq = lambda x: x*x
  
# (deltaT, noisy, lim1, lim2, numTrees, setSizePerSpeaker) depend on the users,
# and are fixed through cross validation for 'n' users.
 
# Let us break up the speech data into manageable chunks.
# deltaT sets the chunk size.
# We need a large chunk size when we have many speakers.
# deltaT also determines the minimum length of time necessary to
# identify the speaker. Thus it is good to keep deltaT < 1 second for 
# real time identification.

deltaT = 0.4   # We need more than 400ms of speech to identify the speaker.

# The combination (deltaT, lim1, lim2) determine the number of features.
# We don't need very high frequency components for human speech.
# Let's get rid of the frequency component at index 0 since this is the DC term.

lim1 = 1; lim2 = 250   # these are frequency indices.

# 'noisy' is required to remove portions of the file with no speech,
# i.e. when the speaker pauses.
# 'noisy' is the fraction of the mean amplitude below which
# there is no speech.
# 'noisy' needs to increased for a noisy background.

noisy = 0.05   # We have a very clean speech sample!

def getData(fil):
  sampFreq, snd = wavfile.read(fil)
  snd = snd/2E15                         # put in range (-1,1)
  duration = snd.shape[0]/sampFreq
  numChunks = int(duration/deltaT)
  sizeChunk = int(len(snd)/numChunks)
  
  # Frequencies.
  freqs  = np.fft.rfftfreq(sizeChunk,deltaT)
  chunksF = []
  for lp in range(0,numChunks):    
    chunk = snd[lp*sizeChunk:(lp+1)*sizeChunk]      # get a chunk of speech.     
    chunksF.append(1E9*np.abs(np.fft.rfft(chunk)))  # take the fft,
                                                    # conveniently normalized.  
  mu = np.mean(chunksF)
  newMean = 0.
  ctr = 0
  for i in range(0,numChunks):
    for j in range(lim1,lim2):
      if abs(chunksF[i][j]) > noisy*mu:         # ignore silent portions.
        newMean += chunksF[i][j]
        ctr += 1

  # Delete portions of the sound file when the user is not speaking.
  mu = newMean/ctr
  zeros = []
  for lp in range(0,numChunks): 
    if np.mean(chunksF[lp]) < noisy*mu:  zeros.append(lp)

  data = []
  ctr = 0
  for i in range(0,numChunks):
    if i in zeros: continue                        # silent part.
    tmp = []
    for j in range(lim1,lim2): tmp.append(chunksF[i][j])
    data.append(tmp)
  
  return data

def writeData(data,outFile):

  # Write the speech waveform to file, for plotting purposes.
  
  l = len(data[0])
  f = open(outFile, "w")
  for i in range(0,len(data)):
    for j in range(0,l):
      f.write(str(i) + " " + str(j) + " " + str(data[i][j]) + "\n")
    f.write("\n")
  f.close()    

def findMeanVar(speechSpeaker):
  
  # Find the mean and standard deviation over time,
  # for each frequency component, for each speaker.
  
  m = len(speechSpeaker)
  n = len(speechSpeaker[0])
  spectraSpeakerMu = []; spectraSpeakerSig = []
  for lp in range(0,n):
    A = [speechSpeaker[i][lp] for i in range(0,m)]
    spectraSpeakerMu.append(np.mean(A))
    spectraSpeakerSig.append(np.std(A))

  return spectraSpeakerMu, spectraSpeakerSig
  
def train(mu,sig):

  # This is the machine learning section.
  # We make use of the fact that speech is not deterministic,
  # i.e. the same person speaking the same words twice will not
  # result in identical waveforms.
  # We therefore simply use the fourier amplitude averaged over time, and
  # the standard deviation, for each frequency bin of interest.
  #
  # We make the assumption that fourier amplitudes are gaussian distributed.
  # Thus, we assume that the amplitude of a fourier mode of a given speaker is
  # in itself, not significant. It is drawn from a distribution defined by (mu,sigma).
  #
  # We need training samples. Since we have a distribution, we can generate a large
  # number of data points consistent with that distribution. 
  # This is our Montecarlo sample.
  # 
  # With this simulated data set, we train a Random Forest Classifier.
  # The Random Forest algorithm has the property that it selects relevant features
  # from the training set, through the computation of the gini impurities.
  # Automatic feature selection and weighting ensures that overfitting is kept
  # to a minimum.
  
  def makeData(n,mu,sig,numSpeakers,numFeatures):

    # Simulate data for numSpeakers.
    X = np.mat(np.zeros((n,numFeatures)))
    y = []
    for i in range(0,n//numSpeakers):
      for j in range(0,numFeatures):
        for k in range(0,numSpeakers):
          d = rd.gauss(mu[k][j],sig[k][j])
          if d < 0.: d = 0.
          X[(numSpeakers*i)+k,j] = d
  
      for k in range(1,numSpeakers+1):
        y.append(k)

    return X,y
  
  
  setSizePerSpeaker = 1000   # size of the training set per speaker.
  numTrees = 100             # number of trees in the forest.

  numSpeakers = len(mu)    
  numFeatures = len(mu[0])        
  trnN = setSizePerSpeaker * numSpeakers # number of samples

  trnX,trnY = makeData(trnN,mu,sig,numSpeakers,numFeatures)      
  return RandomForestClassifier(n_estimators=numTrees, criterion="gini").fit(trnX,trnY)

    
def tstClassifier(tstFolders,clf, length, howMany):

  # Test the classifier!
  
  def getTestSamples(testFolder, numSamples):    

    # Reads from a folder, and returns a set of speech samples.
    sample = [v for v in os.listdir(testFolder) if v <> ".DS_Store"]      
    speechSpeaker = []
    for file in sample:
      lsts = getData(testFolder+file)
      for lst in lsts: speechSpeaker.append(lst)
   
    samples = []
    num = len(speechSpeaker)//numSamples  
    for i in range(0,numSamples):
      samples.append(speechSpeaker[i*num:(i+1)*num])

    return samples
    

  tmpX = []; tmpY = []
  numSpeakers = len(tstFolders)
  numSamples = int(1000*1.4*60./length)    
  ctr = 1
  for tstFolder in tstFolders:
    samples = getTestSamples(tstFolder, numSamples)  
    for sample in samples:
      tmp,junk = findMeanVar(sample)
      tmpX.append(tmp)
      tmpY.append(ctr)
    ctr += 1

  tstX = []; tstY = []

  # Shuffle the arrays.
  randomX = [i for i in range(0,len(tmpX))]
  rd.shuffle(randomX)  
  for val in randomX:
    tstX.append(tmpX[val])
    tstY.append(tmpY[val])
  
  # Let's print some samples if howMany > 0.
  for i in range(0,howMany):
    predicted = clf.predict(tstX[i])[0]
    print "True Speaker: %d, Predicted Speaker: %d" % (tstY[i],predicted)

  # Compute the accuracy.
  yes = [0]*numSpeakers; total = [0]*numSpeakers
  for i in range(0,len(tstX)):
    predicted = clf.predict(tstX[i])[0]
    for k in range(1,numSpeakers+1):
      if predicted == k and tstY[i] == k: yes[k-1] += 1
      if tstY[i] == k: total[k-1] += 1  
    
  # Fraction correctly identified for speakers 1,2,3, . . .
  return yes,total,[1.*yes[k]/total[k] for k in range(0,numSpeakers)]   


def getTrainingSamples(trainingFolders):

  # Each sound file is divided into a number of samples.
  samples = []
  for trainingFolder in trainingFolders:
    tmp = [v for v in os.listdir(trainingFolder) if v <> ".DS_Store"]
    samples.append(tmp)

  # Now samples[0] contains all the files for speaker 1,
  # samples[1] contains all the files for speaker 2, and so on.
  
  speechSpeakers = []
  ctr = 0
  for sample in samples:
    speechSpeaker = []
    for file in sample:
      lsts = getData(trainingFolders[ctr]+file)
      for lst in lsts: speechSpeaker.append(lst)
    speechSpeakers.append(speechSpeaker)
    ctr += 1

  return speechSpeakers
      
      
def main():

  # Let's specify the folders for speaker 1, 2, etc.
  # Training folders have about 5 minutes of data
  trainingFoldersList = [  "wav/training/speaker1/",
                           "wav/training/speaker2/",
                           "wav/training/speaker3/",
                           "wav/training/speaker4/",
                           "wav/training/speaker5/"                       
                        ]

  #Test File folders have about 1.4 minutes worth of data.  
  tstFoldersList = [ "wav/test/speaker1/",
                     "wav/test/speaker2/",
                     "wav/test/speaker3/",
                     "wav/test/speaker4/",
                     "wav/test/speaker5/"                 
                   ]
  # Choose which speakers to work with   
  whichSpeakers = [0,1,2,3,4]            # in this case, all 5. 
  trainingFolders = [trainingFoldersList[speaker] for speaker in whichSpeakers]
  tstFolders      = [tstFoldersList[speaker] for speaker in whichSpeakers]  
  
  # speech[0] contains data for speaker 1, 
  # speech[1] contains data for speaker 2, and so on.     
       
  # Each sound file is divided into a number of samples.
  speech = getTrainingSamples(trainingFolders)
  
  muArray = []; varArray = [] 
  for speaker in speech:   
    # collect mean and standard deviation per frequency.
    mu1,var1 = findMeanVar(speaker)  
    muArray.append(mu1)
    varArray.append(var1)

  # Train a Random Forest Classifier using (mean,variance).
  rfClass = train(muArray,varArray)
  
  # Save the trained classifier.
  #joblib.dump(rfClass, 'rfClass/rfClass.pkl') 

  # Let's output the speech (frequency,time) waveform.  
  #writeData(speech[0],"speaker1.dat")
  #writeData(speech[1],"speaker2.dat")
    
  # Load classifier from disk
  #rfClass = joblib.load("rfClass/rfClass.pkl")    
  
  # Let's test our classifier, and display 15 results.         
  length = 500   # 500 ms.  
  y,t,r = tstClassifier(tstFolders,rfClass,length,15)  # 500 ms of speech.
    
  # We need to make sure that our speech sample size is larger than the chunk size.
  for length in range(2000,int(1000.*deltaT),-50):
    y,t,r = tstClassifier(tstFolders,rfClass,length,0)
    print "%d ms" % length   
    avgAcc = []
    for i in range(0,len(r)):
      print "Speaker: %d  Accuracy: %.2f" % (i+1,r[i])
      avgAcc.append(r[i])
    print "Avg of %d speakers: %.2f" % (len(r),np.mean(avgAcc))
    print "\n"[:-1]  


if __name__ == "__main__":
  main()
  