# Assignment 4

## Introduction

The provided problem statement is to make a model to detect speech. In order to accomplish this, I've used _scikit-learn_ library to make a model and _librosa_ library to extract features from the audio files. The model is trained on the data provided in the [kaggle dataset](https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets). The outline of my approach is as follows:

1. Dataset was downloaded from the above link.
2. Audio data and the corresponding timestamps for silence and speech were extracted from the annotated dataset in the form of `.TextGrid` files.
3. Using _librosa_ library, I was able to capture the MFCCs (Mel Frequency Cepstral Coefficients) of every 20 $ms$ segments of the audio data.
4. The data was split into testing and training data in the ratio of 80:20.
5. The MFCCs were then used as features to train a KNN model using _scikit-learn_ library.
6. Benefits of using KNN are that it is a non-parametric model and it is easy to implement, while simultaneously giving good results.
7. The trained model was saved using _pickle_ library as a `.pkl` file.

## Testing

To test the model, I've provided a main.py file, which can be executed as:

```sh
python3 main.py -i <input_file.wav> -o <output_file.csv>
```

This will produce a `.csv` file with the timestamps of audio segments where silence was there or not (a boolean value). The output file will have the following headers:

```csv
Start Time, End Time, Silence
```
