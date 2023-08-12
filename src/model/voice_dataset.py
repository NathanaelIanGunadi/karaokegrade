from torch.utils.data import DataLoader, Dataset
from feature_extraction import extract_features
from torch import zeros
import logging
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO)

class VoiceDataset(Dataset):
    """
    Dataset for voice data.
    Each data entry is assumed to be a tuple (audio_path, score).
    """

    def __init__(self, data, feature_extraction_func=extract_features, feature_extraction_params=None, training=False):
        """
        Initializes the dataset.

        Parameters:
        - data (list of tuples): List containing tuples of (audio_path, score).
        - feature_extraction_func (callable): Function to use for feature extraction.
        - feature_extraction_params (dict, optional): Additional parameters for feature extraction.
        """
        self.data = data
        self.feature_extraction_func = feature_extraction_func
        self.feature_extraction_params = feature_extraction_params or {}
        self.training = training

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the item at the specified index.

        Parameters:
        - idx (int): Index of the data to fetch.

        Returns:
        - features (torch.Tensor): Extracted audio features.
        - score (float): Corresponding score.
        """
        audio_path, score = self.data[idx]
        try:
            y, sr = librosa.load(audio_path, sr=None)

            if self.training:
                # Apply augmentations
                if np.random.rand() < 0.5:  # 50% chance
                    y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))

                if np.random.rand() < 0.5:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-2, 3))

                if np.random.rand() < 0.5:
                    noise = np.random.randn(len(y))
                    y = y + 0.005 * noise

                if np.random.rand() < 0.5:
                    shift = np.random.randint(sr // 2)
                    y = np.roll(y, shift)

            features = extract_features(y, sr, **self.feature_extraction_params)
        except Exception as e:
            logging.error(f"Error extracting features for {audio_path}. Reason: {e}")
            # Here you can decide whether to return a zero tensor, skip, or do something else
            features = zeros([167, 1])

        return features, score
