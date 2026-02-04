from Audio_Watermarking.algorithms.dct_dwt_svd import *
from Audio_Watermarking.watermarking_tests.attacks import *
from Audio_Watermarking.utils.utils import bit_error_rate
from Audio_Watermarking.watermarking_tests.algorithm_interfaces import *

base_dir = 'Audio_Watermarking/sound_files/audio_dataset/90s'
input_audio = os.path.join(base_dir, '5.wav')
output_audio = os.path.join('Audio_Watermarking/watermarking_tests', 'svd_watermarked.wav')
watermark_bits_int = np.random.randint(0, 2, 256)
watermark_bits = ''.join(watermark_bits_int.astype(str))
frame_size = 1024
step_size = 150.0

warnings.filterwarnings("ignore", category=UserWarning)

alg = DWT_DCT_SVD()

# embed_dct_dwt_svd(input_audio, output_audio, watermark_bits, step_size=step_size, frame_size=frame_size)
alg.embed(input_audio, output_audio, watermark_bits)
cropped_audio = requantize(output_audio, num_bits=6)
extracted_bits = alg.extract(cropped_audio)
extracted_bits = np.frombuffer(extracted_bits.encode("ascii"), dtype=np.uint8) - ord('0')
print(watermark_bits_int ^ extracted_bits)
print(f'BER: {bit_error_rate(watermark_bits, extracted_bits)}')