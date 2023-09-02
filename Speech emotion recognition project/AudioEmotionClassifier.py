# sudo apt install portaudio19-dev
import pandas as pd
import pyaudio
import wave
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tkinter as tk

model = keras.models.load_model('./data/res_model.h5')


def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 4

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wavefile = wave.open("temp.wav", 'wb')
    wavefile.setnchannels(CHANNELS)
    wavefile.setsampwidth(p.get_sample_size(FORMAT))
    wavefile.setframerate(RATE)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    return "temp.wav"


def make_data_noisy(data,random=False,rate=0.050,threshold=0.01):
    """Adds random noise to the input data.

    Args:
        data: Input data.
        random: Whether to apply random noise. If True, rate is calculated randomly using a threshold.
        rate: Magnitude of the noise to be applied.
        threshold: Maximum value of the random rate.

    Returns:
        Augmented data with added noise.

    """
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data


def random_shifting(data,rate=1000):
    """Shifts the input data randomly by a number of samples.

    Args:
        data: Input data.
        rate: Magnitude of the shift.

    Returns:
        Augmented data with shifted samples.
    """
    augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
    augmented_data=np.roll(data,augmented_data)
    return augmented_data


def change_pitch(data,sr,pitch_factor=0.6,random=False):
    """Changes the pitch of the input data.

    Args:
        data: Input data.
        sr: Sampling rate of the input data.
        pitch_factor: Factor to change the pitch by.
        random: Whether to apply random pitch factor. If True, pitch_factor is calculated randomly.

    Returns:
        Augmented data with changed pitch.
    """
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data,sr=sr,n_steps=pitch_factor)


def add_stretch(data,rate=0.7):
    """Stretches the input data by a certain factor.

    Args:
        data: Input data.
        rate: Factor to stretch the data by.

    Returns:
        Augmented data with stretched samples.
    """
    return librosa.effects.time_stretch(data,rate=rate)


def compute_zcr(data, frame_length, hop_length):
    """Computes the zero-crossing rate (ZCR) of the audio signal.

    Args:
        data: Input audio data.
        frame_length: Length of each analysis frame (in samples).
        hop_length: Hop length between consecutive frames (in samples).

    Returns:
        The computed ZCR values as a numpy array.
    """
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def compute_rmse(data, frame_length=2048, hop_length=512):
    """Computes the zero-crossing rate (ZCR) of the audio signal.

    Args:
        data: Input audio data.
        frame_length: Length of each analysis frame (in samples).
        hop_length: Hop length between consecutive frames (in samples).

    Returns:
        The computed ZCR values as a numpy array.
    """
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def compute_mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    """Computes the Mel-frequency cepstral coefficients (MFCC) of the audio signal.

    Args:
        data: Input audio data.
        sr: Sampling rate of the audio data (in Hz).
        frame_length: Length of each analysis frame (in samples).
        hop_length: Hop length between consecutive frames (in samples).
        flatten: Whether to flatten the resulting MFCC matrix to a one-dimensional array.

    Returns:
        The computed MFCC values as a numpy array.
    """
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def identify_features(data, sr, frame_length=2048, hop_length=512):
    """Extracts various audio features from the input audio data.

    Args:
        data: Input audio data.
        sr: Sampling rate of the audio data (in Hz).
        frame_length: Length of each analysis frame (in samples).
        hop_length: Hop length between consecutive frames (in samples).

    Returns:
        A numpy array containing the extracted features.
    """
    result = np.array([])

    result = np.hstack((result,
                        compute_zcr(data, frame_length, hop_length),
                        compute_rmse(data, frame_length, hop_length),
                        compute_mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def get_all_features(path, duration=2.5, offset=0.6):
    """
    Extracts audio features from a given audio file path.

    Args:
    path (str): path to audio file
    duration (float): duration of audio to be loaded in seconds (default=2.5)
    offset (float): offset to start reading audio in seconds (default=0.6)

    Returns:
    np.ndarray: audio features extracted from the audio file
    """
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = identify_features(data, sr)
    audio = np.array(aud)

    audio_with_noise = make_data_noisy(data, random=True)
    aud2 = identify_features(audio_with_noise, sr)
    audio = np.vstack((audio, aud2))

    audio_with_pitch = change_pitch(data, sr, random=True)
    aud3 = identify_features(audio_with_pitch, sr)
    audio = np.vstack((audio, aud3))

    audio_with_pitch1 = change_pitch(data, sr, random=True)
    audio_with_pitch_with_noise = make_data_noisy(audio_with_pitch1, random=True)
    aud4 = identify_features(audio_with_pitch_with_noise, sr)
    audio = np.vstack((audio, aud4))

    return audio


def predict_emotion(file_path):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    aud = identify_features(data, sr)
    audio = np.array(aud)
    # features = get_all_features(file_path)

    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    print(audio.shape)
    audio = np.expand_dims(audio, axis=0)
    print(audio.shape)
    # audio = np.transpose(audio)
    prediction = model.predict(audio)
    print(audio.shape)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = emotion_labels[int(predicted_class)]

    return predicted_label


def on_button_click():
    file_path = record_audio()
    predicted_label = predict_emotion(file_path)
    print("Predicted emotion:", predicted_label)


if __name__ == "__main__":
    root = tk.Tk()
    button = tk.Button(root, text="Start recording", command=on_button_click)
    button.pack()
    root.mainloop()

