from scipy.io import wavfile
import numpy as np
from Audio_Watermarking.utils import utils
import os, warnings
from scipy.fftpack import dct, idct

def embed_dct_spread_spectrum(
    input_wav,
    output_wav,
    watermark_bits,
    alpha=0.1,
    frame_size=1024,
    k1=100,
    k2=400,
    seed=42
):
    sampling_rate, samples = wavfile.read(input_wav)
    is_stereo = len(samples.shape) == 2
    
    channels_to_process = [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    watermarked_channels = []

    for channel_samples in channels_to_process:
        samples = channel_samples.astype(np.float32)
        output = np.copy(samples)

        pn_length = k2 - k1
        np.random.seed(seed)
        pn = np.random.choice([-1, 1], size=pn_length)

        for i, bit in enumerate(watermark_bits):
            start = i * frame_size
            end = start + frame_size

            if end >= len(samples):
                break

            frame = samples[start:end]

            X = dct(frame, norm='ortho')

            # Map bit '0' → -1, bit '1' → +1
            b = 1 if bit == '1' else -1

            # Spread spectrum embedding
            X[k1:k2] += alpha * b * pn
            
            output[start:end] = idct(X, norm='ortho')

        output = np.int16(np.clip(output, -32768, 32767))
        watermarked_channels.append(output)

    if is_stereo:
        watermarked_samples = np.column_stack(watermarked_channels)
    else:
        watermarked_samples = watermarked_channels[0]

    wavfile.write(output_wav, sampling_rate, watermarked_samples)
    print("DCT spread spectrum watermark embedded.")


def extract_dct_spread_spectrum(
    watermarked_wav,
    watermark_length,
    frame_size=1024,
    k1=100,
    k2=400,
    seed=42
):
    _, samples = wavfile.read(watermarked_wav)
    
    is_stereo = len(samples.shape) == 2
    
    channels_to_process = [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    all_extracted_bits = []

    for channel_samples in channels_to_process:
        samples = channel_samples.astype(np.float32)

        pn_length = k2 - k1
        np.random.seed(seed)
        pn = np.random.choice([-1, 1], size=pn_length)

        extracted_bits = ""

        for i in range(watermark_length):
            start = i * frame_size
            end = start + frame_size

            if end >= len(samples):
                break

            frame = samples[start:end]

            X = dct(frame, norm='ortho')

            # Correlation with PN sequence
            corr = np.sum(X[k1:k2] * pn)
            # print(corr)

            extracted_bits += '1' if corr > 0 else '0'
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

# Improved spread spectrum (removes host signal interference)
def embed_dct_iss(
    input_wav,
    output_wav,
    watermark_bits,
    alpha=0.1,
    frame_size=1024,
    k1=100,
    k2=400,
    seed=42
):
    sampling_rate, samples = wavfile.read(input_wav)
    is_stereo = len(samples.shape) == 2
    
    channels_to_process = [samples[:, 0], samples[:, 1]] if is_stereo else [samples]
    watermarked_channels = []

    for channel_samples in channels_to_process:
        samples = channel_samples.astype(np.float32)
        output = np.copy(samples)

        pn_length = k2 - k1
        np.random.seed(seed)
        pn = np.random.choice([-1, 1], size=pn_length)

        for i, bit in enumerate(watermark_bits):
            start = i * frame_size
            end = start + frame_size

            if end >= len(samples):
                break

            frame = samples[start:end]

            X = dct(frame, norm='ortho')

            X_modified = X[k1:k2]

            # Compute lambda         
            lamb = np.dot(X_modified, pn) / (np.dot(X_modified, X_modified) + 1e-9)

            # Map bit '0' → -1, bit '1' → +1
            b = 1 if bit == '1' else -1

            # Spread spectrum embedding
            X[k1:k2] += alpha * (b * pn - lamb * X_modified)
            
            output[start:end] = idct(X, norm='ortho')

        output = np.int16(np.clip(output, -32768, 32767))
        watermarked_channels.append(output)

    if is_stereo:
        watermarked_samples = np.column_stack(watermarked_channels)
    else:
        watermarked_samples = watermarked_channels[0]

    wavfile.write(output_wav, sampling_rate, watermarked_samples)

    print("DCT spread spectrum watermark embedded.")



if __name__ == "__main__":
    # Text Watermark
    base_dir = 'Audio_Watermarking/sound_files'
    input_audio = os.path.join(base_dir, 'Beginning 2.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'I am a secret message!'
    
    warnings.filterwarnings("ignore", category=UserWarning)

    watermark_bits = utils.text_to_bits(watermark)
    print("Watermark bits:", watermark_bits)
    embed_dct_iss(
        input_audio,
        output_audio,
        watermark_bits,
        alpha=0.01
    )
    extracted_bits = extract_dct_spread_spectrum(
        output_audio,
        len(watermark_bits)
    )
    print("Extracted bits:", extracted_bits)
    print("Extracted watermark:", utils.bits_to_text(extracted_bits))