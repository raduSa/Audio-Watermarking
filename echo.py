from scipy.io import wavfile
import numpy as np
import utils
import os

def embed_echo(input_wav, output_wav, watermark_bits,
                        delay_0=50, delay_1=100,
                        alpha=0.5, frame_size=1024):

    sampling_rate, samples = wavfile.read(input_wav)
    # Stereo to mono
    if len(samples.shape) == 2:
        samples = samples[:, 0]

    samples = samples.astype(np.float32)
    output = np.copy(samples)

    for i, bit in enumerate(watermark_bits):
        start = i * frame_size
        end = start + frame_size

        if end >= len(samples):
            break

        frame = samples[start:end]
        delay = delay_0 if bit == '0' else delay_1

        for n in range(delay, frame_size):
            output[start + n] += alpha * frame[n - delay]

    output = np.int16(np.clip(output, -32768, 32767))
    wavfile.write(output_wav, sampling_rate, output)

    print("Non-blind echo watermark embedded.")

def extract_echo_nonblind(original_wav, watermarked_wav,
                          watermark_length,
                          delay_0=50, delay_1=100,
                          frame_size=1024):

    _, original = wavfile.read(original_wav)
    _, watermarked = wavfile.read(watermarked_wav)

    original = original.astype(np.float32)
    watermarked = watermarked.astype(np.float32)

    extracted_bits = ""

    for i in range(watermark_length):
        start = i * frame_size
        end = start + frame_size

        if end >= len(original):
            break

        orig_frame = original[start:end]
        wm_frame = watermarked[start:end]

        # Use correlation for the nonblind version
        corr_0 = np.sum(wm_frame[delay_0:] * orig_frame[:-delay_0])
        corr_1 = np.sum(wm_frame[delay_1:] * orig_frame[:-delay_1])

        host_0 = np.sum(orig_frame[delay_0:] * orig_frame[:-delay_0])
        host_1 = np.sum(orig_frame[delay_1:] * orig_frame[:-delay_1])

        delta_0 = corr_0 - host_0
        delta_1 = corr_1 - host_1

        extracted_bits += '0' if delta_0 > delta_1 else '1'

    return extracted_bits

def extract_echo_blind(watermarked_wav,
                                watermark_length,
                                delay_0=50, delay_1=100,
                                frame_size=1024):
    # For too low alpha values for the encoder might not be detected correctly

    sr, samples = wavfile.read(watermarked_wav)
    samples = samples.astype(np.float32)

    extracted_bits = ""

    window = np.hanning(frame_size)

    for i in range(watermark_length):
        start = i * frame_size
        end = start + frame_size

        if end >= len(samples):
            break

        frame = samples[start:end] * window

        # Cepstrum computation
        spectrum = np.fft.fft(frame)
        log_mag = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.real(np.fft.ifft(log_mag))

        c0 = cepstrum[delay_0] + cepstrum[-delay_0]
        c1 = cepstrum[delay_1] + cepstrum[-delay_1]
        print(c0, c1)

        extracted_bits += '0' if c0 > c1 else '1'

    return extracted_bits



if __name__ == "__main__":
    base_dir = 'Audio-Watermarking'
    input_audio = os.path.join(base_dir, 'Beginning 2.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'I am a secret message!'

    # Text Watermark
    watermark_bits = utils.text_to_bits(watermark)    
    print(watermark_bits)
    embed_echo(input_audio, output_audio, watermark_bits)
    # extracted_watermark_bits = extract_echo_nonblind(input_audio, output_audio, len(watermark_bits))    
    extracted_watermark_bits = extract_echo_blind(output_audio, len(watermark_bits))
    print(extracted_watermark_bits)
    print("Extracted Watermark:", utils.bits_to_text(extracted_watermark_bits))

    # # Audio file Watermark
    # watermark_wav = os.path.join(base_dir, 'bruh.wav')
    # detected_watermark_wav = os.path.join(base_dir, 'detected_bruh.wav')
    # sampling_rate, samples = wavfile.read(watermark_wav)
    # if len(samples.shape) == 2:
    #     samples = samples[:, 0]
    # watermark_bits = utils.audio_to_bits(samples)    
    # print(len(watermark_bits))
    # embed_echo(input_audio, output_audio, watermark_bits)
    # extracted_watermark_bits = extract_echo_nonblind(input_audio, output_audio, len(watermark_bits))
    # print(len(extracted_watermark_bits))
    # extracted_watermark = utils.bits_to_audio(extracted_watermark_bits)
    # wavfile.write(detected_watermark_wav, sampling_rate, np.array(extracted_watermark, dtype=np.int16))
    # print("Extracted Watermark")