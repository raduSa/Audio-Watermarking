import numpy as np
from scipy.io import wavfile

def add_noise(input_wav_path, noise_level=0.005):
    sample_rate, data = wavfile.read(input_wav_path)
    noise = noise_level * np.random.normal(size=data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, -32768, 32767).astype(np.int16)
    wavfile.write(f'noisy_{input_wav_path}', sample_rate, noisy_data)
    
def resample(input_wav_path, target_rate):
    sample_rate, data = wavfile.read(input_wav_path)
    duration = data.shape[0] / sample_rate
    target_length = int(duration * target_rate)
    resampled_data = np.interp(
        np.linspace(0.0, duration, target_length, endpoint=False),
        np.linspace(0.0, duration, data.shape[0], endpoint=False),
        data
    ).astype(np.int16)
    wavfile.write(f'resampled_{input_wav_path}', target_rate, resampled_data)
    