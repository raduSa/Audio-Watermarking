from Audio_Watermarking.utils.utils import *
from Audio_Watermarking.algorithms.lsb import embed_lsb, extract_lsb
from Audio_Watermarking.algorithms.echo import embed_echo, extract_echo_blind, extract_echo_nonblind
from Audio_Watermarking.algorithms.spread_spectrum import embed_dct_iss, extract_dct_spread_spectrum
from Audio_Watermarking.algorithms.qim import embed_qim, extract_qim
from Audio_Watermarking.algorithms.qim_dither import embed_qim_dither, extract_qim_dither
from Audio_Watermarking.algorithms.dwt_qim import embed_dwt_qim, extract_dwt_qim
import numpy as np
import matplotlib.pyplot as plt
import os, warnings


if __name__ == "__main__":
    base_dir = 'Audio_Watermarking/sound_files'
    input_audio = os.path.join(base_dir, 'Biome Fest.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'Fix it from the outside'
    watermark_bits = text_to_bits(watermark)

    warnings.filterwarnings("ignore", category=UserWarning)
    from Audio_Watermarking.watermarking_tests.attacks import resample
    
    resampled_audio = os.path.join(base_dir, f'resampled_16000Hz_{os.path.basename(output_audio)}')

    plt.figure(figsize=(10, 4))
    bers = []
    np.random.seed(42)
    embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=2)
    resample(output_audio, 16000)
    extracted_bits = extract_lsb(resampled_audio, len(watermark_bits), num_lsbs=2)
    ber = bit_error_rate(watermark_bits, extracted_bits)
    bers.append(ber)
    print(f"BER: {ber:.2%}")
    
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    embed_echo(input_audio, output_audio, watermark_bits, alpha=0.8)
    resample(output_audio, 16000)
    extracted_bits = extract_echo_blind(resampled_audio, len(watermark_bits))
    ber = bit_error_rate(watermark_bits, extracted_bits)
    bers.append(ber)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    embed_dct_iss(input_audio, output_audio, watermark_bits, alpha=0.8)
    resample(output_audio, 16000)
    extracted_bits = extract_dct_spread_spectrum(resampled_audio, len(watermark_bits))
    ber = bit_error_rate(watermark_bits, extracted_bits)
    bers.append(ber)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    delta = 10
    frame_size = 2048
    overlap = 512
    
    embed_qim(input_audio, output_audio, watermark_bits, delta=delta, frame_size=frame_size, overlap=overlap)
    resample(output_audio, 16000)
    extracted_bits = extract_qim(resampled_audio, len(watermark_bits), delta=delta, frame_size=frame_size, overlap=overlap)
    ber = bit_error_rate(watermark_bits, extracted_bits)
    bers.append(ber)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    embed_dwt_qim(input_audio, output_audio, watermark_bits, delta=delta, frame_size=frame_size, overlap=overlap)
    resample(output_audio, 16000)
    extracted_bits = extract_dwt_qim(resampled_audio, len(watermark_bits), delta=delta, frame_size=frame_size, overlap=overlap)
    ber = bit_error_rate(watermark_bits, extracted_bits)
    bers.append(ber)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    plt.bar(['LSB', 'Echo', 'Spread-Spectrum', 'QIM', 'DWT-QIM'], bers, color='skyblue')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER after Resampling Attack (16kHz)')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'BER_Resample_Attack.pdf'))
    plt.show()