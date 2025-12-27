import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct, idct
from utils import text_to_bits, bits_to_text, audio_to_bits, bits_to_audio

MID_FREQ_START = 20
MID_FREQ_END = 100

def generate_dither(num_coeffs, delta, seed=420):
    rng = np.random.default_rng(seed)
    return rng.uniform(-delta/4, delta/4, size=num_coeffs)

def quantize(coeff, delta, bit, dither=0.0):
    if bit == 0:
        return delta * np.round((coeff + dither) / delta) - dither
    else:
        return delta * np.round((coeff + dither - delta/2) / delta) + delta/2 - dither

def embed_qim(input_wav, output_wav, watermark, watermark_type, delta, frame_size, overlap, key=420):
    sample_rate, samples = wavfile.read(input_wav)
    if len(samples.shape) == 2:
        samples = samples[:, 0]

    max_samples = samples.shape[0]
    
    if watermark_type == 'text':
        watermark_bits = text_to_bits(watermark)
    elif watermark_type == 'audio':
        _, watermark_samples = wavfile.read(watermark)
        if len(watermark_samples.shape) == 2:
            watermark_samples = watermark_samples[:, 0]
        watermark_bits = audio_to_bits(watermark_samples)
    else:
        raise ValueError("Unsupported watermark type. Use 'text' or 'audio'.")
    watermark_len = len(watermark_bits)

    total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))
    if watermark_len > total_frames:
        raise ValueError("Watermark too large for given frame configuration")

    frames = []
    for i in range(total_frames):
        start = i * (frame_size - overlap)
        end = start + frame_size
        frame = samples[start:end] if end <= max_samples else np.pad(
            samples[start:max_samples], (0, frame_size - (max_samples - start))
        )
        frames.append(dct(frame, norm="ortho"))

    watermark_indices = list(range(MID_FREQ_START, MID_FREQ_END))
    
    for frame_idx, bit in enumerate(watermark_bits):
        dct_coeffs = frames[frame_idx]
        bit = int(bit)
        
        frame_seed = key + frame_idx
        frame_dithers = generate_dither(len(watermark_indices), delta, frame_seed)

        for i, coeff_idx in enumerate(watermark_indices):
            dct_coeffs[coeff_idx] = quantize(dct_coeffs[coeff_idx], delta, bit, frame_dithers[i])
            
        frames[frame_idx] = dct_coeffs

    watermarked_samples = np.zeros(max_samples)
    overlap_count = np.zeros(max_samples)

    for i in range(total_frames):
        start = i * (frame_size - overlap)
        end = min(start + frame_size, max_samples)
        frame = idct(frames[i], norm="ortho")
        watermarked_samples[start:end] += frame[:end - start]
        overlap_count[start:end] += 1

    watermarked_samples /= np.maximum(overlap_count, 1)
    watermarked_samples = np.clip(watermarked_samples, -32768, 32767).astype(np.int16)

    wavfile.write(output_wav, sample_rate, watermarked_samples)
    if watermark_type == 'text':
        print(f"Text watermark embedded successfully. ({len(watermark)} characters)")
    else:
        print(f"Audio watermark embedded successfully. ({len(watermark_samples)} samples)")
    return watermarked_samples

def extract_qim(watermarked_wav, watermark_length, watermark_type, delta, frame_size, overlap, key=420):
    _, samples = wavfile.read(watermarked_wav)
    if len(samples.shape) == 2:
        samples = samples[:, 0]

    max_samples = len(samples)
    total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))

    frames = []
    for i in range(total_frames):
        start = i * (frame_size - overlap)
        end = start + frame_size
        frame = samples[start:end] if end <= max_samples else np.pad(
            samples[start:max_samples], (0, frame_size - (max_samples - start))
        )
        frames.append(dct(frame, norm="ortho"))

    watermark_indices = list(range(MID_FREQ_START, MID_FREQ_END))
    extracted_bits = []

    for frame_idx in range(watermark_length):
        dct_coeffs = frames[frame_idx]
        d0 = 0
        d1 = 0
        
        frame_seed = key + frame_idx
        frame_dithers = generate_dither(len(watermark_indices), delta, frame_seed)

        for i, coeff_idx in enumerate(watermark_indices):
            coeff = dct_coeffs[coeff_idx]
            dither = frame_dithers[i]
            
            q0 = delta * np.round((coeff + dither) / delta) - dither
            q1 = delta * np.round((coeff + dither - delta / 2) / delta) + delta / 2 - dither
            
            d0 += abs(coeff - q0)
            d1 += abs(coeff - q1)
        
        extracted_bits.append('0' if d0 < d1 else '1')

    bits_str = ''.join(extracted_bits)
    
    if watermark_type == 'text':
        return bits_to_text(bits_str)
    elif watermark_type == 'audio':
        return bits_to_audio(bits_str)
    else:
        raise ValueError("Unsupported watermark type. Use 'text' or 'audio'.")


if __name__ == "__main__":
    input_audio = 'Biome Fest.wav'
    output_audio = 'watermarked.wav'
    watermark = 'Fix it from the outside'
    delta = 10
    frame_size = 2048
    overlap = 512
    
    # Text Watermark
    embed_qim(input_audio, output_audio, watermark, 'text', delta, frame_size, overlap)
    extracted_watermark = extract_qim(output_audio, len(text_to_bits(watermark)), 'text', delta, frame_size, overlap)
    print(f"Extracted Watermark: {extracted_watermark}")
    
    # Audio Watermark
    # watermark_audio = 'bruh.wav'
    # embed_qim(input_audio, output_audio, watermark_audio, 'audio', delta, frame_size, overlap)
    # watermark_sample_rate, watermark_samples = wavfile.read(watermark_audio)
    # if len(watermark_samples.shape) == 2:
    #     watermark_len = watermark_samples.shape[0]
    # else:
    #     watermark_len = len(watermark_samples)
    # extracted_audio = extract_qim(output_audio, len(audio_to_bits(watermark_samples)), 'audio', delta, frame_size, overlap)
    # wavfile.write('extracted_watermark.wav', watermark_sample_rate, np.array(extracted_audio, dtype=np.int16))
    # print(f"Extracted audio watermark saved to extracted_watermark.wav")