from abc import ABC, abstractmethod
from Audio_Watermarking.algorithms.lsb import *
from Audio_Watermarking.algorithms.echo import *
from Audio_Watermarking.algorithms.spread_spectrum import *
from Audio_Watermarking.algorithms.qim_dither import *
from Audio_Watermarking.algorithms.dwt_qim import *
from Audio_Watermarking.algorithms.dct_dwt_svd import *

class WatermarkAlgorithm(ABC):    
    def __init__(self, name):
        self.name = name
        self.watermark_length = None

    @abstractmethod
    def embed(self, input_wav, output_wav, watermark_bits):        
        pass

    @abstractmethod
    def extract(self, attacked_wav):        
        pass



class LSB(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("LSB")
        self.num_lsbs = 5

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_lsb(input_wav, output_wav, watermark_bits, num_lsbs=self.num_lsbs)

    def extract(self, attacked_wav):
        return extract_lsb(attacked_wav, self.watermark_length, num_lsbs=self.num_lsbs)

class EchoHiding(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("Echo")
        self.alpha = 0.4

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_echo(input_wav, output_wav, watermark_bits, alpha=self.alpha)

    def extract(self, attacked_wav):
        return extract_echo_blind(attacked_wav, self.watermark_length)

class SpreadSpectrum(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("Spread")
        self.alpha = 0.2

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_dct_iss(input_wav, output_wav, watermark_bits, alpha=self.alpha)

    def extract(self, attacked_wav):
        return extract_dct_spread_spectrum(attacked_wav, self.watermark_length)

class QIMDither(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("QIM")
        self.delta = 10

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_qim_dither(input_wav, output_wav, watermark_bits, delta=self.delta)

    def extract(self, attacked_wav):
        return extract_qim_dither(attacked_wav, self.watermark_length)

class DWT_QIM(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("DWT_QIM")
        self.delta = 10

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_dwt_qim(input_wav, output_wav, watermark_bits, delta=self.delta)

    def extract(self, attacked_wav):
        return extract_dwt_qim(attacked_wav, self.watermark_length)

class DWT_DCT_SVD(WatermarkAlgorithm):
    def __init__(self):
        super().__init__("DWT_DCT_SVD")
        self.step_size = 150

    def embed(self, input_wav, output_wav, watermark_bits):
        self.watermark_length = len(watermark_bits)
        embed_dct_dwt_svd(input_wav, output_wav, watermark_bits)

    def extract(self, attacked_wav):
        return extract_dct_dwt_svd(attacked_wav, self.watermark_length)