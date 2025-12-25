from scipy.io import wavfile
import numpy as np
from utils import text_to_bits, bits_to_text, audio_to_bits, bits_to_audio

def embed_lsb_text(input_wav, output_wav, watermark_text, num_lsbs=1):
    sample_rate, samples = wavfile.read(input_wav)
    print(f"Sample dtype: {samples.dtype}")
    print(f"Bytes per sample: {samples.dtype.itemsize}")
    is_stereo = len(samples.shape) == 2
    max_samples = samples.shape[0] if is_stereo else len(samples)

    watermark_bits = text_to_bits(watermark_text)
    watermark_len = len(watermark_bits)
    
    if watermark_len > max_samples:
        raise ValueError("Watermark too large for this audio file")

    for i in range(watermark_len):
        bit = int(watermark_bits[i])
        if is_stereo:
            for b in range(num_lsbs):
                samples[i, 0] = (samples[i, 0] & ~(1 << b)) | (bit << b)
        else:
            for b in range(num_lsbs):
                samples[i] = (samples[i] & ~(1 << b)) | (bit << b)

    wavfile.write(output_wav, sample_rate, samples)
    
def extract_lsb_text(watermarked_wav, watermark_length, num_lsbs=1):
    sample_rate, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2
    
    bits = []
    for i in range(watermark_length * 8):
        if is_stereo:
            extracted_bits = [(samples[i, 0] >> b) & 1 for b in range(num_lsbs)]
        else:
            extracted_bits = [(samples[i] >> b) & 1 for b in range(num_lsbs)]
        
        bit = 1 if sum(extracted_bits) > num_lsbs / 2 else 0
        bits.append(str(bit))
    
    bits_str = ''.join(bits)
    watermark_text = bits_to_text(bits_str)
    return watermark_text

def embed_lsb_audio(input_wav, output_wav, watermark_audio_wav, num_lsbs=1):
    sample_rate, host_samples = wavfile.read(input_wav)
    _, watermark_samples = wavfile.read(watermark_audio_wav)
    
    is_stereo = len(host_samples.shape) == 2
    max_host_samples = host_samples.shape[0] if is_stereo else len(host_samples)
    
    if len(watermark_samples.shape) == 2:
        watermark_samples = watermark_samples[:, 0]
    
    watermark_bits = audio_to_bits(watermark_samples)
    watermark_len = len(watermark_bits)
    
    if watermark_len > max_host_samples:
        raise ValueError("Watermark audio too large for this audio file")
    
    for i in range(watermark_len):
        bit = int(watermark_bits[i])
        if is_stereo:
            for b in range(num_lsbs):
                host_samples[i, 0] = (host_samples[i, 0] & ~(1 << b)) | (bit << b)
        else:
            for b in range(num_lsbs):
                host_samples[i] = (host_samples[i] & ~(1 << b)) | (bit << b)
    
    wavfile.write(output_wav, sample_rate, host_samples)
    print(f"Audio watermark embedded successfully. ({len(watermark_samples)} samples embedded)")

def extract_lsb_audio(watermarked_wav, watermark_sample_length, num_lsbs=1):
    sample_rate, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2
    
    bits = []
    bit_target = watermark_sample_length * 16
    
    for i in range(bit_target):
        if i < (samples.shape[0] if is_stereo else len(samples)):
            if is_stereo:
                extracted_bits = [(samples[i, 0] >> b) & 1 for b in range(num_lsbs)]
            else:
                extracted_bits = [(samples[i] >> b) & 1 for b in range(num_lsbs)]
            
            bit = 1 if sum(extracted_bits) > num_lsbs / 2 else 0
            bits.append(str(bit))
    
    bits_str = ''.join(bits)
    watermark_audio = bits_to_audio(bits_str)
    return watermark_audio


if __name__ == "__main__":
    input_audio = 'Beginning 2.wav'
    output_audio = 'watermarked.wav'
    watermark = 'I am a secret message!'
    num_lsbs = 3

    # Text Watermark
    # embed_lsb_text(input_audio, output_audio, watermark, num_lsbs=num_lsbs)
    # extracted_watermark = extract_lsb_text(output_audio, len(watermark), num_lsbs=num_lsbs)
    # print("Extracted Watermark:", extracted_watermark)
    
    # Audio Watermark
    watermark_audio = 'bruh.wav'
    embed_lsb_audio(input_audio, output_audio, watermark_audio, num_lsbs=num_lsbs)
    
    _, watermark_samples = wavfile.read(watermark_audio)
    if len(watermark_samples.shape) == 2:
        watermark_len = watermark_samples.shape[0]
    else:
        watermark_len = len(watermark_samples)
    
    extracted_audio = extract_lsb_audio(output_audio, watermark_len, num_lsbs=num_lsbs)
    wavfile.write('extracted_watermark.wav', 48000, np.array(extracted_audio, dtype=np.int16))
    print(f"Extracted audio watermark saved to extracted_watermark.wav")