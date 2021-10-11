import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

speech, rate = librosa.load("gettysburg10.wav",sr=16000)

input_values = tokenizer(speech, return_tensors = 'pt').input_values

logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim =-1)

transcriptions = tokenizer.decode(predicted_ids[0])

print(transcriptions)
