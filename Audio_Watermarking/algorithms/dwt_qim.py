import pywt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct, idct
from Audio_Watermarking.utils.utils import *
import os, warnings

def quantize(coeff, delta=10, bit=0):
    if bit == 0:
        return delta * np.round(coeff / delta)
    else:
        return delta * np.round((coeff - delta / 2) / delta) + delta / 2

def embed_dwt_qim(
    input_wav,
    output_wav,
    watermark_bits,
    delta=10,
    frame_size=1024,
    overlap=512,
    wavelet='db4',
    level=4,
    embed_band=3
):
    sample_rate, samples = wavfile.read(input_wav)
    is_stereo = len(samples.shape) == 2

    channels_to_process = (
        [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    )
    watermarked_channels = []

    for channel_samples in channels_to_process:
        max_samples = len(channel_samples)
        watermark_len = len(watermark_bits)

        total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))
        if watermark_len > total_frames:
            raise ValueError("Watermark too large for given frame configuration")

        frames = []

        # Frame + DWT
        for i in range(total_frames):
            start = i * (frame_size - overlap)
            end = start + frame_size
            frame = (
                channel_samples[start:end]
                if end <= max_samples
                else np.pad(channel_samples[start:max_samples],
                            (0, frame_size - (max_samples - start)))
            )

            coeffs = pywt.wavedec(frame, wavelet, level=level)
            frames.append(coeffs)
        
        for frame_idx, bit in enumerate(watermark_bits):
            bit = int(bit)
            coeffs = frames[frame_idx]

            detail = coeffs[-embed_band]

            # Use a localized coefficient range
            coeff_indices = range(len(detail) // 4, len(detail) // 2)

            for idx in coeff_indices:
                detail[idx] = quantize(detail[idx], delta, bit)

            coeffs[-embed_band] = detail
            frames[frame_idx] = coeffs
        
        watermarked_samples = np.zeros(max_samples)
        overlap_count = np.zeros(max_samples)

        for i in range(total_frames):
            start = i * (frame_size - overlap)
            end = min(start + frame_size, max_samples)

            frame = pywt.waverec(frames[i], wavelet)
            frame = frame[:frame_size]

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

    print(f"DWT-QIM watermark embedded ({len(watermark_bits)} bits)")

def extract_dwt_qim(
    watermarked_wav,
    watermark_length,
    delta=10,
    frame_size=1024,
    overlap=512,
    wavelet='db4',
    level=4,
    embed_band=3
):
    _, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2

    channels_to_process = (
        [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    )
    all_extracted_bits = []

    for channel_samples in channels_to_process:
        max_samples = len(channel_samples)
        total_frames = int(np.ceil((max_samples - overlap) / (frame_size - overlap)))

        frames = []

        # Frame + DWT
        for i in range(total_frames):
            start = i * (frame_size - overlap)
            end = start + frame_size
            frame = (
                channel_samples[start:end]
                if end <= max_samples
                else np.pad(channel_samples[start:max_samples],
                            (0, frame_size - (max_samples - start)))
            )

            frames.append(pywt.wavedec(frame, wavelet, level=level))

        extracted_bits = []
        
        for frame_idx in range(watermark_length):
            coeffs = frames[frame_idx]
            detail = coeffs[-embed_band]

            coeff_indices = range(len(detail) // 4, len(detail) // 2)

            d0 = 0
            d1 = 0

            for idx in coeff_indices:
                coeff = detail[idx]
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
            votes = [int(bits[i]) for bits in all_extracted_bits]
            final_bits.append(str(1 if sum(votes) > len(votes) / 2 else 0))
        return ''.join(final_bits)
    else:
        return ''.join(all_extracted_bits[0])

if __name__ == "__main__":
    base_dir = 'Audio_Watermarking/sound_files'
    input_audio = os.path.join(base_dir, 'Biome Fest.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    extracted_watermark = os.path.join(base_dir, 'extracted_watermark.wav')
    watermark = 'Fix it from the outside'
    delta = 10
    frame_size = 2048
    overlap = 512
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Text Watermark
    watermark_bits = text_to_bits(watermark)
    embed_dwt_qim(input_audio, output_audio, watermark_bits, delta, frame_size, overlap)
    extracted_bits = extract_dwt_qim(output_audio, len(watermark_bits), delta, frame_size, overlap)
    extracted_watermark = bits_to_text(extracted_bits)
    print(f"Extracted Watermark: {extracted_watermark}")