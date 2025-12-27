from scipy.io import wavfile
import numpy as np
import utils
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import utils


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

    # Stereo → mono
    if len(samples.shape) == 2:
        samples = samples[:, 0]

    samples = samples.astype(np.float32)
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
    wavfile.write(output_wav, sampling_rate, output)

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

    samples = samples.astype(np.float32)

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
        print(corr)

        extracted_bits += '1' if corr > 0 else '0'

    return extracted_bits


if __name__ == "__main__":
    # Text Watermark
    base_dir = 'Audio-Watermarking'
    input_audio = os.path.join(base_dir, 'Beginning 2.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'I am a secret message!'

    watermark_bits = utils.text_to_bits(watermark)
    print("Watermark bits:", watermark_bits)
    embed_dct_spread_spectrum(
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
