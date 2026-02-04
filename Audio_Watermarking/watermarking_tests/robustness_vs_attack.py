from Audio_Watermarking.algorithms.lsb import *
from Audio_Watermarking.watermarking_tests.attacks import *
from Audio_Watermarking.utils.utils import bit_error_rate
import os, warnings
import matplotlib.pyplot as plt
from Audio_Watermarking.watermarking_tests.algorithm_interfaces import *
from copy import deepcopy
from Audio_Watermarking.reed_solomon.rs import *

ATTACKS = {
    'noise': {
        'func': lambda p, s: add_noise(p, snr_db=s),
        'strengths': list(range(70, 10, -2))  # dB
    },

    'echo': {
        'func': lambda p, s: add_echo(p, delay_sec=0.3, decay=s),
        'strengths': np.arange(0.001, 0.2, 0.004).tolist()
    },

    'lowpass': {
        'func': lambda p, s: lowpass_filter(p, cutoff_freq=s),
        'strengths': list(range(20000, 1000, -500))
    },

    'highpass': {
        'func': lambda p, s: highpass_filter(p, cutoff_freq=s),
        'strengths': [200, 400, 800, 1200, 2000, 4000]
    },

    'requant': {
        'func': lambda p, s: requantize(p, num_bits=s),
        'strengths': list(range(16, 5, -1))
    },

    'resample': {
        'func': lambda p, s: resample(p, target_rate=s),
        'strengths': [32000, 22050, 16000, 11025, 8000]
    },

    'speed': {
        'func': lambda p, s: speed_change(p, speed_factor=s),
        'strengths': [0.95, 0.98, 1.02, 1.05, 1.1]
    },

    'amplify': {
        'func': lambda p, s: amplify(p, factor=s),
        'strengths': [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
    },

    'compression': {
        'func': lambda p, s: compress(p, bitrate=s),
        'strengths': [f"{b}k" for b in range(300, 100, -20)]
    },

    'cropping': {
        'func': lambda o, w, crop_len : crop_replace_segments(o, w, crop_len),
        'strengths': [10, 100, 200, *range(300, 100000, 10000)]
    }
}


def get_audio_files(dataset_dir):
    return [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith('.wav')
    ]

def evaluate_algorithms_on_attack(
    algorithms,
    attack,
    dataset_dir,
    watermark_bits,
    output_dir,
    use_rs_codes,
    use_log_strength_axis
):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = get_audio_files(dataset_dir)

    plt.figure(figsize=(8, 5))

    orig_watermark_bits = deepcopy(watermark_bits)
    
    orig_files = list()
    for audio_path in audio_files:
        orig_files.append(audio_path)

    if use_rs_codes:
        # The way we have to encode is the following
        # 1. Take the bitstring, pad it and turn it into bytes (store the length, required for decoding)
        # 2. Encode the byte stream into a RS codeword (all codewords will be 255 bytes long)
        # 3. Take the codeword, convert it into a bitstring (2040 bits)
        # 4. Give the bitstring to the embedding algorithm
        watermark_bytes, pad_length = bitstring_to_bytes(watermark_bits)
        watermark_bytes_length = len(watermark_bytes)
        codeword = rs_eval_encode(watermark_bytes, 255)
        watermark_bits = codeword_to_bitstring(codeword)

    for algorithm in algorithms:
        print(f'Testing algortihm: {algorithm.name}')
        ber_curve = list()

        # Apply watermark on whole dataset
        watermarked_files = list()
        for audio_path in audio_files:
            out_path = os.path.join(
                output_dir,
                f'{algorithm.name}_wm_{os.path.basename(audio_path)}'
            )
            algorithm.embed(audio_path, out_path, watermark_bits)
            watermarked_files.append(out_path)

        # Sweep attack strengths
        for i, strength in enumerate(ATTACKS[attack]['strengths']):
            bers = list()

            for j, wm_path in enumerate(watermarked_files):
                print(f"Testing {i * len(watermarked_files) + j} \
                      /{len(watermarked_files) * len(ATTACKS[attack]['strengths'])}")
                if attack == 'cropping':
                    attacked_path = ATTACKS[attack]['func'](orig_files[j], wm_path, strength)
                else:
                    attacked_path = ATTACKS[attack]['func'](wm_path, strength)

                extracted_bits = algorithm.extract(attacked_path)

                if use_rs_codes:
                    # The way we extract the watermark is the same as described above, just inverted order
                    codeword = bitstring_to_codeword(extracted_bits)
                    decoded_bytes = rs_decode(codeword, 255, watermark_bytes_length)
                    extracted_bits = bytes_to_bitstring(decoded_bytes, pad_length)

                ber = bit_error_rate(orig_watermark_bits, extracted_bits)
                bers.append(ber)

            ber_curve.append(np.mean(bers))

        plt.plot(
            ATTACKS[attack]['strengths'] \
            if attack != 'compression' \
            else [int(val[:-1]) for val in ATTACKS[attack]['strengths']],
            ber_curve,
            marker='o',
            label=algorithm.name
        )

    plt.xlabel('Attack Strength')
    plt.ylabel('Average BER')
    plt.title(f'Robustness vs Attack Strength ({attack})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if use_log_strength_axis:
        plt.xscale('log')

    plot_path = os.path.join(output_dir, f"{attack}_robustness{'_rs' if use_rs_codes else ''}.pdf")
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    algorithms = [        
        LSB(),
        EchoHiding(),
        SpreadSpectrum(),
        QIMDither(),
        DWT_QIM(),
        DWT_DCT_SVD()
    ]

    watermark_bits = np.random.randint(0, 2, 256)
    watermark_bits = ''.join(watermark_bits.astype(str))
    # When using rs codes, the 90s dataset should be used (as 30s is too few samples)
    audio_dataset = f'Audio_Watermarking/sound_files/audio_dataset/90s'
    test_outputs = f'Audio_Watermarking/watermarking_tests/testing_helper'

    evaluate_algorithms_on_attack(
        algorithms=algorithms,
        attack='requant',
        dataset_dir=audio_dataset,
        watermark_bits=watermark_bits,
        output_dir=test_outputs,
        use_rs_codes=True,
        use_log_strength_axis=False
    )
