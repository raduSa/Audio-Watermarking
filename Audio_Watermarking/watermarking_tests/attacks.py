import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
from Audio_Watermarking.algorithms.dct_dwt_svd import *

def read_audio(input_wav_path):
    sample_rate, data = wavfile.read(input_wav_path)
    return sample_rate, data.astype(np.float64)

def write_audio(output_path, sample_rate, data):
    data = np.clip(data, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, data)

import os
import numpy as np

def add_echo(input_wav_path, delay_sec=0.5, decay=0.5):
    sample_rate, data = read_audio(input_wav_path)
    
    delay_samples = int(delay_sec * sample_rate)
    length = len(data)
    
    output = np.copy(data).astype(float)
    
    if delay_samples < length:
        remaining_space = length - delay_samples
        
        output[delay_samples:] += decay * data[:remaining_space]
    
    peak_in = np.max(np.abs(data))
    peak_out = np.max(np.abs(output))
    if peak_out > 0:
        output *= peak_in / peak_out
    
    echoed_audio = os.path.join(os.path.dirname(input_wav_path), f'echoed_{os.path.basename(input_wav_path)}')
    write_audio(echoed_audio, sample_rate, output)
    return echoed_audio

def add_noise(input_wav_path, snr_db=50):
    sample_rate, data = read_audio(input_wav_path)
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*data.shape)
    noisy_data = data + noise
    noisy_audio = os.path.join(os.path.dirname(input_wav_path), f'noisy_{os.path.basename(input_wav_path)}')
    write_audio(noisy_audio, sample_rate, noisy_data)
    return noisy_audio    
    
def amplify(input_wav_path, factor):
    sample_rate, data = read_audio(input_wav_path)
    amplified_data = data * factor
    amplified_audio = os.path.join(os.path.dirname(input_wav_path), f'amplified_{os.path.basename(input_wav_path)}')
    write_audio(amplified_audio, sample_rate, amplified_data)
    return amplified_audio

def compress(input_wav_path, bitrate="64k"):
    audio = AudioSegment.from_wav(input_wav_path)
    compressed_path = os.path.join(os.path.dirname(input_wav_path), f'compressed_{os.path.basename(input_wav_path).replace(".wav", ".mp3")}')
    audio.export(compressed_path, format="mp3", bitrate=bitrate)
    compressed_audio = os.path.join(os.path.dirname(input_wav_path), f'decompressed_{os.path.basename(input_wav_path)}')
    AudioSegment.from_mp3(compressed_path).export(compressed_audio, format="wav")
    return compressed_audio

    
def crop_replace_segments(
    original_wav_path,
    watermarked_wav_path,    
    crop_len=1000,    
):
    sr_orig, original = read_audio(original_wav_path)
    sr_wm, watermarked = read_audio(watermarked_wav_path)

    if sr_orig != sr_wm:
        raise ValueError("Original and watermarked sample rates must match")

    length = len(original)

    if crop_len >= length:
        raise ValueError("crop_len must be smaller than signal length")
    
    attacked = np.copy(watermarked)
    
    for position in ['front', 'middle', 'end']:
        if position == "front":
            start = 0
        elif position == "middle":
            start = length // 2 - crop_len // 2
        elif position == "end":
            start = length - crop_len        

        end = start + crop_len
        
        attacked[start:end] = original[start:end]

    output_path = os.path.join(
        os.path.dirname(watermarked_wav_path),
        f"crop_{os.path.basename(watermarked_wav_path)}"
    )

    write_audio(output_path, sr_wm, attacked)
    return output_path

def lowpass_filter(input_wav_path, cutoff_freq, num_taps=101):
    sample_rate, audio = read_audio(input_wav_path)
    is_stereo = len(audio.shape) == 2
    fc = cutoff_freq / sample_rate
    if fc <= 0 or fc >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency!")
    n = np.arange(num_taps)
    h = 2 * fc * np.sinc(2 * fc * (n - (num_taps - 1) / 2))
    h *= np.blackman(num_taps)  
    h /= np.sum(h)
    
    if is_stereo:
        filtered_audio = np.zeros_like(audio)
        for channel in range(audio.shape[1]):
            filtered_audio[:, channel] = np.convolve(audio[:, channel], h, mode='same')
    else:
        filtered_audio = np.convolve(audio, h, mode='same')
    
    lowpass_audio = os.path.join(os.path.dirname(input_wav_path), f'lowpass_{os.path.basename(input_wav_path)}')
    write_audio(lowpass_audio, sample_rate, filtered_audio)
    return lowpass_audio
    
def highpass_filter(input_wav_path, cutoff_freq, num_taps=101):
    sample_rate, audio = read_audio(input_wav_path)
    is_stereo = len(audio.shape) == 2
    fc = cutoff_freq / sample_rate
    if fc <= 0 or fc >= 1:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency!")
    n = np.arange(num_taps) - (num_taps - 1) / 2
    h_lp = 2 * fc * np.sinc(2 * fc * n)
    h_lp *= np.blackman(num_taps)
    h_lp /= np.sum(h_lp)
    h_hp = -h_lp
    h_hp[(num_taps - 1) // 2] += 1
    
    if is_stereo:
        filtered_audio = np.zeros_like(audio)
        for channel in range(audio.shape[1]):
            filtered_audio[:, channel] = np.convolve(audio[:, channel], h_hp, mode='same')
    else:
        filtered_audio = np.convolve(audio, h_hp, mode='same')
    
    highpass_audio = os.path.join(os.path.dirname(input_wav_path), f'highpass_{os.path.basename(input_wav_path)}')
    write_audio(highpass_audio, sample_rate, filtered_audio)
    return highpass_audio

def requantize(input_wav_path, num_bits=8):
    if num_bits <= 0 or num_bits > 16:
        raise ValueError("Number of bits must be between 1 and 16!")
    
    sample_rate, data = read_audio(input_wav_path)
    max_val = 2**(num_bits - 1) - 1
    min_val = -2**(num_bits - 1)
    
    quantized_data = np.round(data / 32768 * max_val)
    quantized_data = np.clip(quantized_data, min_val, max_val)
    quantized_data = (quantized_data / max_val) * 32768
    
    requantized_audio = os.path.join(os.path.dirname(input_wav_path), f'requantized_{os.path.basename(input_wav_path)}')
    write_audio(requantized_audio, sample_rate, quantized_data)
    return requantized_audio

def resample(input_wav_path, target_rate):
    if target_rate <= 0:
        raise ValueError("Target rate must be positive!")
    
    sample_rate, data = read_audio(input_wav_path)
    if target_rate >= sample_rate:
        raise ValueError("Target rate must be lower than original sample rate for downsampling attack!")

    is_stereo = len(data.shape) == 2
    ratio = target_rate / sample_rate
    target_length = max(1, int(data.shape[0] * ratio))
    
    # Downsample with anti-aliasing
    if is_stereo:
        downsampled = np.zeros((target_length, data.shape[1]))
        for channel in range(data.shape[1]):
            downsampled[:, channel] = signal.resample(data[:, channel], target_length)
    else:
        downsampled = signal.resample(data, target_length)
    
    # Upsample back to original length
    original_length = data.shape[0]
    if is_stereo:
        upsampled = np.zeros((original_length, data.shape[1]))
        for channel in range(data.shape[1]):
            upsampled[:, channel] = signal.resample(downsampled[:, channel], original_length)
    else:
        upsampled = signal.resample(downsampled, original_length)
    
    resampled_audio = os.path.join(os.path.dirname(input_wav_path), f'resampled_{os.path.basename(input_wav_path)}')
    write_audio(resampled_audio, sample_rate, upsampled)    
    return resampled_audio
