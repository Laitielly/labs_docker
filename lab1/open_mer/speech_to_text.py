import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def get_the_text() -> np.array:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    model.eval()

    signal, sr = librosa.load('utt.mp3', sr=16000)
    input_values = processor(signal, return_tensors="pt", sampling_rate=sr, padding="longest").input_values

    logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return np.array(transcription)