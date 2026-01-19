# TODO

- Add W&B integration behind `wandb.enabled` (run metadata, metrics, artifacts).
- Add integration tests for the full pipeline on synthetic HDF5 inputs.
- Add CLI smoke tests for `train.py` and `visualize.py`.
- Add dataset sharding option for large feature sets.
- Expand dataset metadata tracking (court surface, indoor/outdoor, player info) into manifests.
- Add optional augmentation hooks for future data sources and label generation.
- Document inference-time downsampling strategy separate from training preproc.




- test YOLO preproc hyperparams, particularly yolo size on quality of model outputs
- fine tune YOLO-n on YOLO-L outputs


- test automating data collection with model self-dataset generation


