from Audio_Watermarking.algorithms.lsb import *
from Audio_Watermarking.watermarking_tests.attacks import *
from Audio_Watermarking.utils.utils import bit_error_rate
import os, warnings
import matplotlib.pyplot as plt
from Audio_Watermarking.watermarking_tests.algorithm_interfaces import *
from copy import deepcopy
from Audio_Watermarking.reed_solomon.rs import *
from Audio_Watermarking.watermarking_tests.robustness_vs_attack import get_audio_files

import os
from scipy.io import wavfile

def copy_wav(input_path, output_path):
    sample_rate, samples = wavfile.read(input_path)
    wavfile.write(output_path, sample_rate, samples)
    
def save_watermarked_and_unwatermarked(
    algorithms,
    dataset_dir,
    watermark_bits,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = get_audio_files(dataset_dir)
    
    for algorithm in algorithms:        
        algorithm_outputs = os.path.join(output_dir, algorithm.name)
        os.makedirs(algorithm_outputs, exist_ok=True)
                
        # Pick only 7 audio files at random from dataset
        picked_idxs = np.random.choice(len(audio_files), size=7, replace=False)
        for idx in picked_idxs:
            audio_path = audio_files[idx]
            audio_file_outputs = os.path.join(algorithm_outputs, os.path.basename(audio_path))
            os.makedirs(audio_file_outputs, exist_ok=True)

            # Copy unwatermarked
            out_path = os.path.join(
                audio_file_outputs,
                f"unwatermarked.wav"
            )
            copy_wav(audio_path, out_path)

            # Write watermarked
            out_path = os.path.join(
                audio_file_outputs,
                f"watermarked.wav"
            )
            algorithm.embed(audio_path, out_path, watermark_bits)

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
    watermark_bits = ''.join(watermark_bits.astype(str))
    audio_dataset = f'Audio_Watermarking/sound_files/audio_dataset/30s'
    outputs = f'Audio_Watermarking/watermarking_tests/AXB'

    save_watermarked_and_unwatermarked(
        algorithms=algorithms,
        dataset_dir=audio_dataset,
        watermark_bits=watermark_bits,
        output_dir=outputs,
    )
