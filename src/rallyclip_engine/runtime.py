from __future__ import annotations

from rallyclip_core.contracts import RuntimeDeps


def load_runtime_deps() -> RuntimeDeps:
    """Load heavy analysis dependencies on demand."""
    import numpy as np  # noqa: WPS433
    from extraction.pose_extractor import PoseExtractor  # noqa: WPS433
    from features.feature_engineer import FeatureEngineer  # noqa: WPS433
    from infer import (  # noqa: WPS433
        HeatmapDecodeConfig,
        decode_heatmap_segments,
        extract_segments_from_binary,
        gaussian_filter1d,
        hysteresis_threshold,
        load_model_from_checkpoint,
        load_scaler_asset,
        run_multitrack_windowed_inference_onnx_stream,
        run_windowed_inference_average_onnx_stream,
        run_windowed_inference_average_torch_stream,
        write_segments_csv,
    )
    from preprocessing.data_preprocessor import DataPreprocessor  # noqa: WPS433
    from runtime.device import apply_pose_device  # noqa: WPS433
    from segmentation.segment import segment_video  # noqa: WPS433

    return RuntimeDeps(
        np=np,
        PoseExtractor=PoseExtractor,
        DataPreprocessor=DataPreprocessor,
        FeatureEngineer=FeatureEngineer,
        load_scaler_asset=load_scaler_asset,
        load_model_from_checkpoint=load_model_from_checkpoint,
        run_windowed_inference_average_onnx_stream=run_windowed_inference_average_onnx_stream,
        run_windowed_inference_average_torch_stream=run_windowed_inference_average_torch_stream,
        gaussian_filter1d=gaussian_filter1d,
        hysteresis_threshold=hysteresis_threshold,
        extract_segments_from_binary=extract_segments_from_binary,
        write_segments_csv=write_segments_csv,
        segment_video=segment_video,
        apply_pose_device=apply_pose_device,
        run_multitrack_windowed_inference_onnx_stream=run_multitrack_windowed_inference_onnx_stream,
        decode_heatmap_segments=decode_heatmap_segments,
        heatmap_decode_config_cls=HeatmapDecodeConfig,
    )
