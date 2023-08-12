import librosa
import torch
import numpy as np
import logging

# Constants
DEFAULT_MFCC_COEFFICIENTS = 20  # Just an example, librosa's default is 20

logging.basicConfig(level=logging.INFO)


def extract_features(audio_input, sr=None, use_mel=True, use_mfcc=True, use_chroma=True, use_contrast=True,
                     mfcc_coefficients=DEFAULT_MFCC_COEFFICIENTS):
    """
    Extract audio features from the given audio file.

    Parameters:
    - audio_path (str): Path to the audio file.
    - use_mel, use_mfcc, use_chroma, use_contrast (bool): Flags to determine which features to extract.

    Returns:
    - torch.Tensor: Extracted features.
    """
    try:
        if isinstance(audio_input, str):  # if it's a path
            y, sr = librosa.load(audio_input, sr=None)
        else:
            y = audio_input

        y = (y - np.mean(y)) / np.std(y)

        features = []

        if use_mel:
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            features.append(mel)

        if use_mfcc:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_coefficients)
            features.append(mfcc)

        if use_chroma:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(chroma)

        if use_contrast:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.append(contrast)

        return torch.tensor(np.vstack(features))

    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise e  # Raise the exception so you can handle it in your calling function or let the error stop execution if it's crucial.
