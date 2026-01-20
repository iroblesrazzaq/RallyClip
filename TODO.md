# TODO

- Add W&B integration behind `wandb.enabled` (run metadata, metrics, artifacts).
- Add integration tests for the full pipeline on synthetic HDF5 inputs.
- Add CLI smoke tests for `train.py` and `visualize.py`.
- Add dataset sharding option for large feature sets.
- Expand dataset metadata tracking (court surface, indoor/outdoor, player info) into manifests.
- Add optional augmentation hooks for future data sources and label generation.
- Document inference-time downsampling strategy separate from training preproc.




- test YOLO preproc hyperparams, particularly yolo size on quality of model outputs


- test automating data collection with model self-dataset generation
- SwingVision data generation: csv vs video matching from cut and whole video for csv generation?


- improve postprocessing: train another model with IOU?
- another LSTM?


- try LSTM with attention, other architectures, hyperparams. 

- optimize inference: push YOLO inference cost as far down as possible
- minimize downsampled frames
- introduce linear interpolation for features?

- fine tune YOLO-n on YOLO-L outputs, reduced imgsz

- optimize inference with batching, other stuff?
- video resolution: if we can reduce video res for smaller yolo inference, would be optimal: potentially reduce imgsz



- add aggresiveness slider for postprocessing tuning of to have more sensitive to inclusion vs not



- document failure cases for court detector, maybe make more robust?
