import pywt
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct, idct
from Audio_Watermarking.utils.utils import *
import os, warnings

def embed_dct_dwt_svd(input_wav, output_wav, watermark_bits, frame_size = 2048):
    sample_rate, samples = wavfile.read(input_wav)
    is_stereo = len(samples.shape) == 2
    channels_to_process = ([samples[:, 0], samples[:, 1]] if is_stereo else [samples])
    watermarked_channels = []

    for channel_samples in channels_to_process:
        frames = len(channel_samples) // frame_size
        if len(watermark_bits) > frames:
            raise ValueError("Watermark too large for given frame configuration!")
        
        for i in range(min(frames, len(watermark_bits))):
            start = i * frame_size
            end = start + frame_size
            frame = channel_samples[start:end]
            
            coeffs = pywt.wavedec(frame, 'haar', level=2)
            D2 = coeffs[1]
            D1 = coeffs[2]
            X = np.concatenate([D1, D2, D2])
            Y = dct(X, norm='ortho')
            N = int(np.sqrt(frame_size))
            C = Y[:N*N].reshape(N, N)
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            
            S_marked = S.copy()
            Q = 150.0
            quantized = np.round(S[0] / Q) * Q
            bit = int(watermark_bits[i])
            if bit == 0:
                S_marked[0] = quantized + Q / 4
            else:
                S_marked[0] = quantized + 3 * Q / 4
                
            C_marked = U @ np.diag(S_marked) @ Vt
            Y_marked = Y.copy()
            Y_marked[:N*N] = C_marked.flatten()
            X_marked = idct(Y_marked, norm='ortho')
            D1_marked = X_marked[:len(D1)]
            D2_first = X_marked[len(D1):len(D1) + len(D2)]
            D2_second = X_marked[len(D1) + len(D2):len(D1) + 2 * len(D2)]
            D2_marked = (D2_first + D2_second) / 2
            coeffs[1] = D2_marked
            coeffs[2] = D1_marked
            watermarked_frame = pywt.waverec(coeffs, 'haar')
            channel_samples[start:end] = watermarked_frame[:frame_size]

        watermarked_channels.append(channel_samples)
        
    if is_stereo:
        watermarked_samples = np.column_stack(watermarked_channels)
    else:
        watermarked_samples = watermarked_channels[0]
        
    wavfile.write(output_wav, sample_rate, np.int16(np.clip(watermarked_samples, -32768, 32767)))
    print("DCT-DWT-SVD watermark embedded.")

def extract_dct_dwt_svd(watermarked_wav, watermark_length, frame_size = 2048):
    _, samples = wavfile.read(watermarked_wav)
    is_stereo = len(samples.shape) == 2

    channels_to_process = ([samples[:, 0], samples[:, 1]] if is_stereo else [samples])
    all_extracted_bits = []

    for channel_samples in channels_to_process:
        samples = channel_samples.astype(np.float32)
        extracted_bits = ""

        for i in range(min(watermark_length, len(samples) // frame_size)):
            start = i * frame_size
            end = start + frame_size

            if end >= len(samples):
                break

            frame = samples[start:end]
            coeffs = pywt.wavedec(frame, 'haar', level=2)
            D2 = coeffs[1]
            D1 = coeffs[2]
            X = np.concatenate([D1, D2, D2])
            Y = dct(X, norm='ortho')
            N = int(np.sqrt(frame_size))
            C = Y[:N*N].reshape(N, N)
            U, S, Vt = np.linalg.svd(C, full_matrices=False)
            
            Q = 150.0
            R = S[0] % Q
            if R < Q / 2:
                extracted_bits += '0'
            else:
                extracted_bits += '1'

        all_extracted_bits.append(extracted_bits)
        
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
    output_audio = os.path.join(base_dir, 'svd_watermarked.wav')
    extracted_watermark = os.path.join(base_dir, 'extracted_watermark.wav')
    watermark = 'Fix it from the outside'
    frame_size = 4096
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Text Watermark
    watermark_bits = text_to_bits(watermark)
    embed_dct_dwt_svd(input_audio, output_audio, watermark_bits, frame_size)
    extracted_bits = extract_dct_dwt_svd(output_audio, len(watermark_bits), frame_size)
    extracted_watermark = bits_to_text(extracted_bits)
    print(f"Extracted Watermark: {extracted_watermark}")