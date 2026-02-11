from typing import Optional

from .encoder import Encoder
from .encoder_geofdn import EncoderGeoFDN, EncoderGeoFDNCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_geofdn import EncoderVisualizerGeoFDN

ENCODERS = {
    "geofdn": (EncoderGeoFDN, EncoderVisualizerGeoFDN),
}

EncoderCfg = EncoderGeoFDNCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
