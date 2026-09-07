from .inference import (
    extract_segments_from_binary,
    gaussian_filter1d,
    hysteresis_threshold,
    load_scaler_asset,
    load_model_from_checkpoint,
    run_windowed_inference_average_onnx,
    run_windowed_inference_average,
    run_windowed_inference_average_stream,
    run_windowed_inference_average_onnx_stream,
    run_windowed_inference_average_torch_stream,
    run_multitrack_windowed_inference_stream,
    run_multitrack_windowed_inference_onnx_stream,
    run_multitrack_windowed_inference_torch_stream,
    write_segments_csv,
)
from .heatmap_decode import (
    HeatmapDecodeConfig,
    decode_heatmap_segments,
    decode_hybrid,
    decode_peakpair,
)

__all__ = [
    "TennisPointLSTM",
    "extract_segments_from_binary",
    "gaussian_filter1d",
    "hysteresis_threshold",
    "load_scaler_asset",
    "load_model_from_checkpoint",
    "run_windowed_inference_average_onnx",
    "run_windowed_inference_average",
    "run_windowed_inference_average_stream",
    "run_windowed_inference_average_onnx_stream",
    "run_windowed_inference_average_torch_stream",
    "run_multitrack_windowed_inference_stream",
    "run_multitrack_windowed_inference_onnx_stream",
    "run_multitrack_windowed_inference_torch_stream",
    "write_segments_csv",
    "HeatmapDecodeConfig",
    "decode_heatmap_segments",
    "decode_hybrid",
    "decode_peakpair",
]




def __getattr__(name):
    # Lazy: .model imports torch, which is optional at runtime (ONNX path).
    if name == "TennisPointLSTM":
        from .model import TennisPointLSTM

        return TennisPointLSTM
    raise AttributeError(name)
