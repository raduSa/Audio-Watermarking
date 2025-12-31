import numpy as np
import os
from scipy.io import wavfile
from pydub import AudioSegment

def read_audio(input_wav_path):
    sample_rate, data = wavfile.read(input_wav_path)
    return sample_rate, data.astype(np.float64)

def write_audio(output_path, sample_rate, data):
    data = np.clip(data, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, data)

def add_echo(input_wav_path, delay_sec=0.5, decay=0.5):
    sample_rate, data = read_audio(input_wav_path)
    delay_samples = int(delay_sec * sample_rate)
    is_stereo = len(data.shape) == 2
    
    if is_stereo:
        output = np.zeros((len(data) + delay_samples, data.shape[1]))
        output[:len(data)] = data
        output[delay_samples:delay_samples + len(data)] += decay * data
    else:
        output = np.zeros(len(data) + delay_samples)
        output[:len(data)] = data
        output[delay_samples:delay_samples + len(data)] += decay * data
    
    peak_in = np.max(np.abs(data))
    peak_out = np.max(np.abs(output))
    if peak_out > 0:
        output *= peak_in / peak_out
    
    echoed_audio = os.path.join(os.path.dirname(input_wav_path), f'echoed_{os.path.basename(input_wav_path)}')
    write_audio(echoed_audio, sample_rate, output)

def add_noise(input_wav_path, snr_db=50):
    sample_rate, data = read_audio(input_wav_path)
    signal_power = np.mean(data**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*data.shape)
    noisy_data = data + noise
    noisy_audio = os.path.join(os.path.dirname(input_wav_path), f'noisy_{os.path.basename(input_wav_path)}')
    write_audio(noisy_audio, sample_rate, noisy_data)
    
def amplify(input_wav_path, factor):
    sample_rate, data = read_audio(input_wav_path)
    amplified_data = data * factor
    amplified_data /= np.max(np.abs(amplified_data)) / np.max(np.abs(data))
    amplified_audio = os.path.join(os.path.dirname(input_wav_path), f'amplified_{os.path.basename(input_wav_path)}')
    write_audio(amplified_audio, sample_rate, amplified_data)

def compress(input_wav_path, bitrate="64k"):
    audio = AudioSegment.from_wav(input_wav_path)
    compressed_path = os.path.join(os.path.dirname(input_wav_path), f'compressed_{os.path.basename(input_wav_path).replace(".wav", ".mp3")}')
    audio.export(compressed_path, format="mp3", bitrate=bitrate)
    compressed_audio = os.path.join(os.path.dirname(input_wav_path), f'decompressed_{os.path.basename(input_wav_path)}')
    AudioSegment.from_mp3(compressed_path).export(compressed_audio, format="wav")
    
def crop(input_wav_path, start_sec, end_sec):
    sample_rate, data = read_audio(input_wav_path)
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    cropped = data[start_sample:end_sample]
    cropped_audio = os.path.join(os.path.dirname(input_wav_path), f'cropped_{os.path.basename(input_wav_path)}')
    write_audio(cropped_audio, sample_rate, cropped)
    
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
    
def mix_to_mono(input_wav_path):
    sample_rate, data = read_audio(input_wav_path)
    if len(data.shape) == 2:
        mono_data = data.mean(axis=1)
        mono_audio = os.path.join(os.path.dirname(input_wav_path), f'mono_{os.path.basename(input_wav_path)}')
        write_audio(mono_audio, sample_rate, mono_data)
    else:
        print("Audio is already mono!")
    
def resample(input_wav_path, target_rate):
    sample_rate, data = read_audio(input_wav_path)
    is_stereo = len(data.shape) == 2
    duration = data.shape[0] / sample_rate
    target_length = int(duration * target_rate)
    
    # Downsample
    if is_stereo:
        downsampled = np.zeros((target_length, data.shape[1]))
        for channel in range(data.shape[1]):
            downsampled[:, channel] = np.interp(
                np.linspace(0.0, duration, target_length, endpoint=False),
                np.linspace(0.0, duration, data.shape[0], endpoint=False),
                data[:, channel]
            )
    else:
        downsampled = np.interp(
            np.linspace(0.0, duration, target_length, endpoint=False),
            np.linspace(0.0, duration, data.shape[0], endpoint=False),
            data
        )
    
    # Upsample back
    if is_stereo:
        upsampled = np.zeros((data.shape[0], data.shape[1]))
        for channel in range(data.shape[1]):
            upsampled[:, channel] = np.interp(
                np.linspace(0.0, duration, data.shape[0], endpoint=False),
                np.linspace(0.0, duration, target_length, endpoint=False),
                downsampled[:, channel]
            )
    else:
        upsampled = np.interp(
            np.linspace(0.0, duration, data.shape[0], endpoint=False),
            np.linspace(0.0, duration, target_length, endpoint=False),
            downsampled
        )
    
    resampled_audio = os.path.join(os.path.dirname(input_wav_path), f'resampled_{os.path.basename(input_wav_path)}')
    write_audio(resampled_audio, sample_rate, upsampled)