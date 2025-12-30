import numpy as np
from scipy.io import wavfile
from Audio_Watermarking.utils.utils import *

def embed_lsb(input_wav, output_wav, watermark_bits, num_lsbs=1):
    sample_rate, host_samples = wavfile.read(input_wav)
    is_stereo = len(host_samples.shape) == 2
    max_host_samples = host_samples.shape[0] if is_stereo else len(host_samples)
    
    watermark_len = len(watermark_bits)
    
    if watermark_len > max_host_samples:
        raise ValueError(f"Watermark too large! Need {watermark_len} samples, have {max_host_samples}")
    
    for i in range(watermark_len):
        bit = int(watermark_bits[i])
        
        if is_stereo:
            for b in range(num_lsbs):
                host_samples[i, 0] = (host_samples[i, 0] & ~(1 << b)) | (bit << b)
        else:
            for b in range(num_lsbs):
                host_samples[i] = (host_samples[i] & ~(1 << b)) | (bit << b)
    
    wavfile.write(output_wav, sample_rate, host_samples)
    print(f"Watermark embedded: {watermark_len} bits using {num_lsbs} LSB(s)")


def extract_lsb(watermarked_wav, num_bits, num_lsbs=1):
    _, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2
    
    max_samples = samples.shape[0] if is_stereo else len(samples)
    
    if num_bits > max_samples:
        raise ValueError(f"Cannot extract {num_bits} bits from {max_samples} samples!")
    
    bits = []
    for i in range(num_bits):
        if is_stereo:
            extracted_bits = [(samples[i, 0] >> b) & 1 for b in range(num_lsbs)]
        else:
            extracted_bits = [(samples[i] >> b) & 1 for b in range(num_lsbs)]
        
        # Majority voting when using multiple LSBs
        bit = 1 if sum(extracted_bits) > num_lsbs / 2 else 0
        bits.append(str(bit))
    
    bits_str = ''.join(bits)
    print(f"Extracted: {num_bits} bits using {num_lsbs} LSB(s)")
    
    return bits_str


if __name__ == "__main__":
    input_audio = 'Beginning 2.wav'
    output_audio = 'watermarked.wav'
    num_lsbs = 3
    
    # Text Watermark
    watermark_text = 'I love signal processing!'
    watermark_bits = text_to_bits(watermark_text)
    print(f"Original text: '{watermark_text}'")
    print(f"Converted to {len(watermark_bits)} bits")
    
    embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=num_lsbs)
    extracted_bits = extract_lsb(output_audio, len(watermark_bits), num_lsbs=num_lsbs)
    extracted_text = bits_to_text(extracted_bits)
    print(f"Extracted text: '{extracted_text}'")
    print(f"Match: {watermark_text == extracted_text}")
    
    # Audio Watermark
    watermark_audio = 'bruh.wav'
    watermark_sr, watermark_samples = wavfile.read(watermark_audio)
    if len(watermark_samples.shape) == 2:
        watermark_samples = watermark_samples[:, 0]
    print(f"Watermark audio: {len(watermark_samples)} samples at {watermark_sr} Hz")
    watermark_bits = audio_to_bits(watermark_samples)
    print(f"Converted to {len(watermark_bits)} bits")
    
    embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=num_lsbs)
    extracted_bits = extract_lsb(output_audio, len(watermark_bits), num_lsbs=num_lsbs)
    extracted_samples = bits_to_audio(extracted_bits)
    wavfile.write('extracted_watermark.wav', watermark_sr, np.array(extracted_samples, dtype=np.int16))
    print(f"Extracted audio: {len(extracted_samples)} samples")
    print(f"Saved as extracted_watermark.wav")