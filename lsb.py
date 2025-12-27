import numpy as np
from scipy.io import wavfile
from utils import text_to_bits, bits_to_text, audio_to_bits, bits_to_audio

def embed_lsb(input_wav, output_wav, watermark, watermark_type, num_lsbs=1):
    sample_rate, host_samples = wavfile.read(input_wav)
    is_stereo = len(host_samples.shape) == 2
    max_host_samples = host_samples.shape[0] if is_stereo else len(host_samples)
    
    if watermark_type == 'text':
        watermark_bits = text_to_bits(watermark)
    elif watermark_type == 'audio':
        _, watermark_samples = wavfile.read(watermark)
        if len(watermark_samples.shape) == 2:
            watermark_samples = watermark_samples[:, 0]
        watermark_bits = audio_to_bits(watermark_samples)
    else:
        raise ValueError("Unsupported watermark type!")
    watermark_len = len(watermark_bits)
    
    if watermark_len > max_host_samples:
        raise ValueError("Watermark too large for this audio file!")

    for i in range(watermark_len):
        bit = int(watermark_bits[i])
        if is_stereo:
            for b in range(num_lsbs):
                host_samples[i, 0] = (host_samples[i, 0] & ~(1 << b)) | (bit << b)
        else:
            for b in range(num_lsbs):
                host_samples[i] = (host_samples[i] & ~(1 << b)) | (bit << b)

    wavfile.write(output_wav, sample_rate, host_samples)
    if watermark_type == 'text':
        print(f"Text watermark embedded successfully. ({len(watermark)} characters)")
    else:
        print(f"Audio watermark embedded successfully. ({len(watermark_samples)} samples)")

def extract_lsb(watermarked_wav, watermark_length, watermark_type, num_lsbs=1):
    _, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2
    
    bits = []
    if watermark_type == 'text':
        for i in range(watermark_length * 8):
            if is_stereo:
                extracted_bits = [(samples[i, 0] >> b) & 1 for b in range(num_lsbs)]
            else:
                extracted_bits = [(samples[i] >> b) & 1 for b in range(num_lsbs)]
            
            bit = 1 if sum(extracted_bits) > num_lsbs / 2 else 0
            bits.append(str(bit))
        
        bits_str = ''.join(bits)
        watermark = bits_to_text(bits_str)
        
    elif watermark_type == 'audio':
        bit_target = watermark_length * 16
        for i in range(bit_target):
            if i < (samples.shape[0] if is_stereo else len(samples)):
                if is_stereo:
                    extracted_bits = [(samples[i, 0] >> b) & 1 for b in range(num_lsbs)]
                else:
                    extracted_bits = [(samples[i] >> b) & 1 for b in range(num_lsbs)]
                
                bit = 1 if sum(extracted_bits) > num_lsbs / 2 else 0
                bits.append(str(bit))
        
        bits_str = ''.join(bits)
        watermark = bits_to_audio(bits_str)
    else:
        raise ValueError("Unsupported watermark type!")
    
    return watermark

if __name__ == "__main__":
    input_audio = 'Beginning 2.wav'
    output_audio = 'watermarked.wav'
    watermark = 'I am a secret message!'
    num_lsbs = 3

    # Text Watermark
    # embed_lsb(input_audio, output_audio, watermark, 'text', num_lsbs=num_lsbs)
    # extracted_watermark = extract_lsb(output_audio, len(watermark), watermark_type='text', num_lsbs=num_lsbs)
    # print("Extracted Watermark:", extracted_watermark)
    
    # Audio Watermark
    watermark_audio = 'bruh.wav'
    embed_lsb(input_audio, output_audio, watermark_audio, watermark_type='audio', num_lsbs=num_lsbs)
    
    watermark_sample_rate, watermark_samples = wavfile.read(watermark_audio)
    if len(watermark_samples.shape) == 2:
        watermark_len = watermark_samples.shape[0]
    else:
        watermark_len = len(watermark_samples)
    
    extracted_audio = extract_lsb(output_audio, watermark_len, watermark_type='audio', num_lsbs=num_lsbs)
    wavfile.write('extracted_watermark.wav', watermark_sample_rate, np.array(extracted_audio, dtype=np.int16))
    print(f"Extracted audio watermark saved to extracted_watermark.wav")