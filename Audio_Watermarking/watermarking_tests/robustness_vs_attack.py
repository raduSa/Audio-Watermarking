from Audio_Watermarking.algorithms.lsb import *
from Audio_Watermarking.watermarking_tests.attacks import *
from Audio_Watermarking.utils.utils import bit_error_rate
import os, warnings
import matplotlib.pyplot as plt
from Audio_Watermarking.watermarking_tests.algorithm_interfaces import *

ATTACKS = {
    "noise": {
        "func": lambda p, s: add_noise(p, snr_db=s),
        "strengths": [50, 45, 40, 35, 30, 25, 20, 15, 10]
    },
    "echo": {
        "func": lambda p, s: add_echo(p, delay_sec=0.3, decay=s),
        "strengths": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
    "lowpass": {
        "func": lambda p, s: lowpass_filter(p, cutoff_freq=s),
        "strengths": [16000, 14000, 12000, 10000, 8000, 6000, 4000]
    },
    "requant": {
        "func": lambda p, s: requantize(p, num_bits=s),
        "strengths": [16, 14, 12, 10, 8, 6]
    },
    "resample": {
        "func": lambda p, s: resample(p, target_rate=s),
        "strengths": [32000, 22050, 16000, 11025, 8000]
    },
    "speed": {
        "func": lambda p, s: speed_change(p, speed_factor=s),
        "strengths": [0.95, 0.98, 1.02, 1.05, 1.1]
    }
}

def get_audio_files(dataset_dir):
    return [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith(".wav")
    ]

def evaluate_algorithms_on_attack(
    algorithms,
    attack,
    dataset_dir,
    watermark_bits,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = get_audio_files(dataset_dir)

    plt.figure(figsize=(8, 5))

    for algorithm in algorithms:
        ber_curve = []

        # Apply watermark on whole dataset
        watermarked_files = []
        for audio_path in audio_files:
            out_path = os.path.join(
                output_dir,
                f"{algorithm.name}_wm_{os.path.basename(audio_path)}"
            )
            algorithm.embed(audio_path, out_path, watermark_bits)
            watermarked_files.append(out_path)

        # Sweep attack strengths
        for strength in ATTACKS[attack]["strengths"]:
            bers = []

            for wm_path in watermarked_files:
                attacked_path = deepcopy(wm_path)
                ATTACKS["func"](attacked_path, strength)

                extracted = algorithm.extract(attacked_path)
                ber = bit_error_rate(watermark_bits, extracted)
                bers.append(ber)

            ber_curve.append(np.mean(bers))

        plt.plot(
            ATTACKS["strengths"],
            ber_curve,
            marker="o",
            label=algorithm.name
        )

    plt.xlabel("Attack Strength")
    plt.ylabel("Average BER")
    plt.title(f"Robustness vs Attack Strength ({attack})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{attack}_robustness.png")
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    algorithms = [
        LSB(),
        EchoHiding(),
        SpreadSpectrum(),
        QIMDither(),
        DWT_QIM(),
        DWT_DCT_SVD(),
    ]

    watermark_bits = np.random.randint(0, 2, 256)

    evaluate_algorithms_on_attack(
        algorithms=algorithms,
        attack='noise'
        dataset_dir='dataset_wavs',
        watermark_bits=watermark_bits,
        output_dir='evaluation_results'
    )
