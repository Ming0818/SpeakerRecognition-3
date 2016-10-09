# Speaker Recognition

Data comes from the LibriSpeech ASR corpus

http://www.openslr.org/12/

File: train-clean-100.tar.gz (6.3G)(training set of 100 hours "clean" speech) 

5 male speakers and 5 female speakers.

Deep Feed Forward Neural Network built with PyBrain: 

Mean accuracy (over 10 speakers) is 92% with 1 second of voice data, compared to 10% for random guessing. With 3 seconds of voice data, the mean accuracy is 99%. With 200 milliseconds of data, the accuracy is 74%.
