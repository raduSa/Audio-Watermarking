from Audio_Watermarking.utils.utils import *
from Audio_Watermarking.algorithms.lsb import embed_lsb, extract_lsb
from Audio_Watermarking.algorithms.echo import embed_echo, extract_echo_nonblind
from Audio_Watermarking.algorithms.spread_spectrum import embed_dct_iss, extract_dct_spread_spectrum
from Audio_Watermarking.algorithms.qim import embed_qim, extract_qim
from Audio_Watermarking.algorithms.qim_dither import embed_qim_dither, extract_qim_dither
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    base_dir = 'Audio_Watermarking/sound_files'
    input_audio = os.path.join(base_dir, 'Biome Fest.wav')
    output_audio = os.path.join(base_dir, 'watermarked.wav')
    watermark = 'Fix it from the outside'
    watermark_bits = text_to_bits(watermark)

    from Audio_Watermarking.watermarking_tests.attacks import add_noise

    np.random.seed(42)
    embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=2)
    add_noise(output_audio)
    extracted_bits = extract_lsb(f'noisy_{output_audio}', len(watermark_bits), num_lsbs=2)
    ber = bit_error_rate(watermark_bits, extracted_bits)
    print(f"BER: {ber:.2%}")
    
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    embed_echo(input_audio, output_audio, watermark_bits, alpha=0.8)
    add_noise(output_audio)
    extracted_bits = extract_echo_nonblind(input_audio, f'noisy_{output_audio}', len(watermark_bits))
    ber = bit_error_rate(watermark_bits, extracted_bits)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    embed_dct_iss(input_audio, output_audio, watermark_bits, alpha=0.8)
    add_noise(output_audio)
    extracted_bits = extract_dct_spread_spectrum(f'noisy_{output_audio}', len(watermark_bits))
    ber = bit_error_rate(watermark_bits, extracted_bits)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    delta = 10
    frame_size = 2048
    overlap = 512
    
    embed_qim(input_audio, output_audio, watermark_bits, delta=delta, frame_size=frame_size, overlap=overlap)
    add_noise(output_audio)
    extracted_bits = extract_qim(f'noisy_{output_audio}', len(watermark_bits), delta=delta, frame_size=frame_size, overlap=overlap)
    ber = bit_error_rate(watermark_bits, extracted_bits)
    print(f"BER: {ber:.2%}")
    extracted_text = bits_to_text(extracted_bits)
    print(f"Original: '{watermark}'")
    print(f"Extracted: '{extracted_text}'")
    
    # snrs_lsb = []
    # for num_lsbs in range(1, 6):
    #     embed_lsb(input_audio, output_audio, watermark_bits, num_lsbs=num_lsbs)
    #     snrs_lsb.append(SNR(input_audio, output_audio))
    
    # plt.figure()
    # plt.plot(range(1, 6), snrs_lsb, marker='o')
    # plt.title('LSB Watermarking: SNR vs Number of LSBs Used')
    # plt.xlabel('Number of LSBs Used')
    # plt.ylabel('SNR (dB)')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()