import unittest
from models.ptq.layers import QLinear
from config import Config
import torch

class TestQLinear(unittest.TestCase):
    def test_qlinear(self):
        linear = QLinear(3,
                5,
                quant=False,
                calibrate=False,
                bit_type=cfg.BIT_TYPE_W,
                calibration_mode=cfg.CALIBRATION_MODE_W,
                observer_str=cfg.OBSERVER_W,
                quantizer_str=cfg.QUANTIZER_W)
        
        x = torch.randn(2, 3)
        y = linear(x)
        print('x: ', x)
        print('y: ', y)


if __name__ == "__main__":

    cfg = Config()
    unittest.main()
