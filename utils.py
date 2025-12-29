def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def audio_to_bits(samples):
    bits = []
    for sample in samples:
        sample = int(sample)
        unsigned_sample = sample if sample >= 0 else 65536 + sample
        bits.append(format(unsigned_sample, '016b'))
    return ''.join(bits)

def bits_to_audio(bits):
    samples = []
    for i in range(0, len(bits), 16):
        if i + 16 <= len(bits):
            byte = bits[i:i+16]
            sample = int(byte, 2)
            if sample >= 32768:
                sample -= 65536
            samples.append(sample)
    return samples

import numpy as np
from scipy.io import wavfile

def SNR(original_signal, watermarked_signal):
    if isinstance(original_signal, str):
        _, original = wavfile.read(original_signal)
        if len(original.shape) == 2:
            original = original[:, 0]
    else:
        original = np.array(original_signal)
    
    if isinstance(watermarked_signal, str):
        _, watermarked = wavfile.read(watermarked_signal)
        if len(watermarked.shape) == 2:
            watermarked = watermarked[:, 0]
    else:
        watermarked = np.array(watermarked_signal)
    
    noise = watermarked - original
    
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)

def normalized_correlation(original, watermarked):
    return np.dot(original, watermarked) / (np.linalg.norm(original) * np.linalg.norm(watermarked))

def bit_error_rate(original_bits, extracted_bits):
    if len(original_bits) != len(extracted_bits):
        raise ValueError("Bit sequences must have the same length")
    errors = sum(int(o) ^ int(e) for o, e in zip(original_bits, extracted_bits))
    return errors / len(original_bits)

def character_error_rate(original_text, extracted_text):
    errors = sum(o != e for o, e in zip(original_text, extracted_text))
    return errors / len(original_text)