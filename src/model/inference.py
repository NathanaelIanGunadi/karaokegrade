from model import VoiceGrader
from feature_extraction import extract_features
from post_processing import moving_average
import torch

def infer(model_path, audio_files, input_size=167):
    """
    Load the trained VoiceGrader model and predict scores for given audio files.

    Parameters:
    - model_path (str): Path to the saved model.
    - audio_files (list): List of paths to audio files.
    - input_size (int): Input size for the model. Default is 167.

    Returns:
    - list: Smoothed scores for the audio files.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VoiceGrader(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predicted_scores = []

    for audio_file in audio_files:
        features = extract_features(audio_file).unsqueeze(0).to(device).float()
        with torch.no_grad():
            score = model(features)
            score = torch.clamp(score, min=0, max=100)
            predicted_scores.append(score.item())

    # Now apply post-processing to the predicted scores
    smoothed_scores = moving_average(predicted_scores, window_size=1)

    return smoothed_scores

if __name__ == "__main__":
    model_path = "best_model.pth"
    audio_files = ["../../audio/test/test2.wav"]  # list of audio files to infer on

    scores = infer(model_path, audio_files)
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    print(scores)
