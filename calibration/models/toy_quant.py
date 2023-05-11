from torch import nn

from .layers_quant import trunc_normal_
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear

__all__ = ['toy']

class Toy(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 input_quant=False,
                 output_quant=False,
                 cfg=None):
        super().__init__()

        self.input_quant = input_quant
        self.output_quant = output_quant
        self.cfg = cfg
        if input_quant:
            self.qact_input = QAct(bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)
        
        self.head = QLinear(in_features=in_features,
                             out_features=out_features,
                             bias=bias,
                             bit_type=cfg.BIT_TYPE_W,
                             calibration_mode=cfg.CALIBRATION_MODE_W,
                             observer_str=cfg.OBSERVER_W,
                             quantizer_str=cfg.QUANTIZER_W)
        if output_quant:
            self.act_out = QAct(bit_type=cfg.BIT_TYPE_A,
                                calibration_mode=cfg.CALIBRATION_MODE_A,
                                observer_str=cfg.OBSERVER_A,
                                quantizer_str=cfg.QUANTIZER_A)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward(self, x):
        if self.input_quant:
            x = self.qact_input(x)
        x = self.head(x)
        if self.output_quant:
            x = self.act_out(x)
        return x
    
def toy(in_features,
        out_features,
        input_quant=False,
        output_quant=False,
        cfg=None,
        **kwargs):
    
    """
    Args: 
        in_features: int, input feature dimension
        out_features: int, output feature dimension
        input_quant: bool, whether to quantize input
        output_quant: bool, whether to quantize output (whether to add a last activation layer)
        cfg: config, configuration for quantization
    """
    model = Toy(in_features=in_features,
                out_features=out_features,
                input_quant=input_quant,
                output_quant=output_quant,
                cfg=cfg,
                **kwargs)
    return model