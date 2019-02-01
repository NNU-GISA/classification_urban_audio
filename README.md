# Urban Audio Classification

## Goal

The goal of this project is to predict the category of raw audio files by extracting different numeric representations of audio and testing various classification algorithms. Dimensionality reduction will play a large role in the success of the correct classification.

## Base Data Set

The Urban Sound dataset is composed of 8,732 labeled audio files from 10 different classes. The [academic-sourced dataset](https://urbansounddataset.weebly.com/urbansound8k.html) seeks to serve urban sound researchers. To this end, the 10 labeled classes focus on sounds that contribute to urban noise pollution. It was compiled by examining the top noise complaints in NYC: think things like jackhammers, gun shots, etc.

## Feature Extraction

In order to run the data through classification algorithms I first have to extract relevant audio features. There's significant academic writing on this topic. After a brief survey, I've chosen to extract the following numerical based representations of audio:

1. The waveforms Spectral constant
2. The waveforms Chromagram
3. The waverforms Mel-frequency cepstral coefficients

The [librosa](https://librosa.github.io/librosa/feature.html) python module will handle the extraction details. The feature extraction loops though the audio and  calculates values at equal intervals in the audio file. I skimmed through some music based analysis papers, and am attempting to apply similar methods to these short city field recordings. One challenge area is that these methods may not translate as effectively to these shorter audio clips. The end result will be 62-length-vectors for each audio file, i.e. an 8732 x 62 matrix, plus one column for the labels.

## Other Notes

The dataset authors created this dataset by splicing longer recordings. E.g. the data set is 4 second clips, but it was created by splicing say a 50 minute recording at different points. One result of this is that the data cannot be shuffled. Otherwise, the models could learn the characteristics of the longer recording and thus you'd leak information post-shuffle. Thankfully, they've corrected for this, and pre-computed 10-folds to work with that won't suffer from this problem. So the goal will be to run 10-fold CV and average how you do on the training blocks.
