import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct, idct
from Audio_Watermarking.utils.utils import *

MID_FREQ_START = 20
MID_FREQ_END = 100

def quantize(coeff, delta=10, bit=0):
    if bit == 0:
        return delta * np.round(coeff / delta)
    else:
        return delta * np.round((coeff - delta / 2) / delta) + delta / 2

def embed_qim(input_wav, output_wav, watermark_bits, delta=10, frame_size=1024, overlap=512):
    sample_rate, samples = wavfile.read(input_wav)
    is_stereo = len(samples.shape) == 2
    
    channels_to_process = [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    watermarked_channels = []
    
    for channel_samples in channels_to_process:
        max_samples = len(channel_samples)
        watermark_len = len(watermark_bits)

        total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))
        if watermark_len > total_frames:
            raise ValueError("Watermark too large for given frame configuration")

        frames = []
        for i in range(total_frames):
            start = i * (frame_size - overlap)
            end = start + frame_size
            frame = channel_samples[start:end] if end <= max_samples else np.pad(
                channel_samples[start:max_samples], (0, frame_size - (max_samples - start))
            )
            frames.append(dct(frame, norm="ortho"))

        watermark_indices = list(range(MID_FREQ_START, MID_FREQ_END))

        for frame_idx, bit in enumerate(watermark_bits):
            dct_coeffs = frames[frame_idx]
            bit = int(bit)

            for coeff_idx in watermark_indices:
                dct_coeffs[coeff_idx] = quantize(dct_coeffs[coeff_idx], delta, bit)

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
        watermarked_channels.append(watermarked_samples)
    
    if is_stereo:
        stereo_output = np.column_stack((watermarked_channels[0], watermarked_channels[1]))
        wavfile.write(output_wav, sample_rate, stereo_output)
    else:
        wavfile.write(output_wav, sample_rate, watermarked_channels[0])
    
    print(f"Watermark embedded successfully. ({len(watermark_bits)} bits, {len(channels_to_process)} channel(s))")
    return watermarked_channels[0]

def extract_qim(watermarked_wav, watermark_length, delta=10, frame_size=1024, overlap=512):
    _, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2
    
    channels_to_process = [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    all_extracted_bits = []
    
    for channel_samples in channels_to_process:
        max_samples = len(channel_samples)
        total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))

        frames = []
        for i in range(total_frames):
            start = i * (frame_size - overlap)
            end = start + frame_size
            frame = channel_samples[start:end] if end <= max_samples else np.pad(
                channel_samples[start:max_samples], (0, frame_size - (max_samples - start))
            )
            frames.append(dct(frame, norm="ortho"))

        watermark_indices = list(range(MID_FREQ_START, MID_FREQ_END))
        extracted_bits = []

        for frame_idx in range(watermark_length):
            dct_coeffs = frames[frame_idx]
            d0 = 0
            d1 = 0

            for coeff_idx in watermark_indices:
                coeff = dct_coeffs[coeff_idx]
                q0 = delta * np.round(coeff / delta)
                q1 = delta * np.round((coeff - delta / 2) / delta) + delta / 2
                d0 += abs(coeff - q0)
                d1 += abs(coeff - q1)

            extracted_bits.append('0' if d0 < d1 else '1')
        
        all_extracted_bits.append(extracted_bits)
    
    # Majority voting if stereo
    if is_stereo:
        final_bits = []
        for i in range(watermark_length):
            bit_votes = [int(channel_bits[i]) for channel_bits in all_extracted_bits]
            final_bits.append(str(1 if sum(bit_votes) > len(bit_votes) / 2 else 0))
        return ''.join(final_bits)
    else:
        return ''.join(all_extracted_bits[0])


if __name__ == "__main__":
    input_audio = 'Biome Fest.wav'
    output_audio = 'watermarked.wav'
    watermark = 'Fix it from the outside'
    delta = 10
    frame_size = 2048
    overlap = 512
    
    # Text Watermark
    watermark_bits = text_to_bits(watermark)
    embed_qim(input_audio, output_audio, watermark_bits, delta, frame_size, overlap)
    extracted_bits = extract_qim(output_audio, len(watermark_bits), delta, frame_size, overlap)
    extracted_watermark = bits_to_text(extracted_bits)
    print(f"Extracted Watermark: {extracted_watermark}")
    
    # Audio Watermark
    # watermark_audio = 'bruh.wav'
    # watermark_sample_rate, watermark_samples = wavfile.read(watermark_audio)
    # if len(watermark_samples.shape) == 2:
    #     watermark_samples = watermark_samples[:, 0]
    # watermark_bits = audio_to_bits(watermark_samples)
    # embed_qim(input_audio, output_audio, watermark_bits, delta, frame_size, overlap)
    # extracted_bits = extract_qim(output_audio, len(watermark_bits), delta, frame_size, overlap)
    # extracted_audio = bits_to_audio(extracted_bits)
    # wavfile.write('extracted_watermark.wav', watermark_sample_rate, np.array(extracted_audio, dtype=np.int16))
    # print(f"Extracted audio watermark saved to extracted_watermark.wav")