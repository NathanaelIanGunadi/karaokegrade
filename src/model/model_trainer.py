from model import *
from voice_dataset import *
from torch.nn.utils.rnn import pad_sequence
import json


def collate_fn(batch):
    sequences, labels = zip(*batch)

    # Find the maximum length of sequences in this batch
    max_length = max([s.size(1) for s in sequences])

    # Pad each sequence to the max length
    padded_sequences = torch.stack([torch.cat([s, torch.zeros(s.size(0), max_length - s.size(1))], dim=1) for s in sequences])

    return padded_sequences, torch.tensor(labels)




def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def main(config):
    train_data = [
        ("../../audio/train/hello.wav", 93),
        ("../../audio/train/sugar.wav", 81),
        ("../../audio/train/easy_on_me.wav", 91),
        ("../../audio/train/kaijuu_no_hanauta.wav", 88),
        ("../../audio/train/short_kaijuu.wav", 77),
        ("../../audio/train/short_love_in_the_dark.wav", 89),
        ("../../audio/train/kirari.wav", 91),
        ("../../audio/train/test1.wav", 68)
        # ... add more training samples ...
    ]
    val_data = [
        ("../../audio/val/love_in_the_dark.wav", 90),
        ("../../audio/val/dryflower.wav", 88)
        # ... add validation samples ...
    ]

    train_dataset = VoiceDataset(train_data, training=True)
    val_dataset = VoiceDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grader = VoiceGrader(config["input_size"], config["hidden_size"], device=device)

    optimizer = torch.optim.Adam(grader.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    trainer = Trainer(grader, optimizer, criterion, num_epochs=10, device=device)

    # Train the model using the training and validation loaders
    trainer.train_model(train_loader, val_loader)


if __name__ == "__main__":
    configs = load_config("../../configs/trainer_config.json")
    main(configs)
