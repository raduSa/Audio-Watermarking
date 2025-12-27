import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

def add_noise(input_wav_path, noise_level=0.005):
    sample_rate, data = wavfile.read(input_wav_path)
    noise = noise_level * np.random.normal(size=data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, -32768, 32767).astype(np.int16)
    wavfile.write(f'noisy_{input_wav_path}', sample_rate, noisy_data)
    
def resample(input_wav_path, target_rate):
    sample_rate, data = wavfile.read(input_wav_path)
    is_stereo = len(data.shape) == 2
    duration = data.shape[0] / sample_rate
    target_length = int(duration * target_rate)
    
    if is_stereo:
        resampled_data = np.zeros((target_length, data.shape[1]), dtype=np.int16)
        for channel in range(data.shape[1]):
            resampled_data[:, channel] = np.interp(
                np.linspace(0.0, duration, target_length, endpoint=False),
                np.linspace(0.0, duration, data.shape[0], endpoint=False),
                data[:, channel]
            ).astype(np.int16)
    else:
        resampled_data = np.interp(
            np.linspace(0.0, duration, target_length, endpoint=False),
            np.linspace(0.0, duration, data.shape[0], endpoint=False),
            data
        ).astype(np.int16)
    
    wavfile.write(f'resampled_{input_wav_path}', target_rate, resampled_data)
    
def compress(input_wav_path, bitrate="64k"):
    audio = AudioSegment.from_wav(input_wav_path)
    compressed_path = f'compressed_{input_wav_path.replace(".wav", ".mp3")}'
    audio.export(compressed_path, format="mp3", bitrate=bitrate)
    AudioSegment.from_mp3(compressed_path).export(f'decompressed_{input_wav_path}', format="wav")

def lowpass_filter(input_wav_path, cutoff_freq, num_taps=101):
    sample_rate, audio = wavfile.read(input_wav_path)
    is_stereo = len(audio.shape) == 2
    fc = cutoff_freq / (sample_rate / 2)
    if fc <= 0 or fc >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency!")
    n = np.arange(num_taps)
    h = fc * np.sinc(fc * (n - (num_taps - 1) / 2))
    h *= np.blackman(num_taps)  
    h /= np.sum(h)
    
    if is_stereo:
        filtered_audio = np.zeros_like(audio)
        for channel in range(audio.shape[1]):
            filtered_audio[:, channel] = np.convolve(audio[:, channel], h, mode='same').astype(np.int16)
    else:
        filtered_audio = np.convolve(audio, h, mode='same').astype(np.int16)
    
    wavfile.write(f'lowpass_{input_wav_path}', sample_rate, filtered_audio)
    
def highpass_filter(input_wav_path, cutoff_freq, num_taps=101):
    sample_rate, audio = wavfile.read(input_wav_path)
    is_stereo = len(audio.shape) == 2
    fc = cutoff_freq / (sample_rate / 2)
    if fc <= 0 or fc >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency!")
    n = np.arange(num_taps)
    h = -fc * np.sinc(fc * (n - (num_taps - 1) / 2))
    h[(num_taps - 1) // 2] += 1
    h *= np.blackman(num_taps)
    h /= np.sum(h)
    
    if is_stereo:
        filtered_audio = np.zeros_like(audio)
        for channel in range(audio.shape[1]):
            filtered_audio[:, channel] = np.convolve(audio[:, channel], h, mode='same').astype(np.int16)
    else:
        filtered_audio = np.convolve(audio, h, mode='same').astype(np.int16)
    
    wavfile.write(f'highpass_{input_wav_path}', sample_rate, filtered_audio)