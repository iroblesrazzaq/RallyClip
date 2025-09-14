# file to run inference on an entire feature npz file, then apply postprocessing steps
# to output final start_time,end_time csv file
# %%
import numpy as np
import torch
from lstm_model_arch import TennisPointLSTM
from scipy.ndimage import gaussian_filter1d

"""
My test.py file is my current evaluation file. however, it just looks at sequences, not the whole video. 
I need to further establish my post processing pipeline. The final output of my pipeline should be 
a csv of start_time, end_times, which i can compare to the annotated targets. 
For now, we will use the same gaussian smoothing and hysteresis filtering that we're using in the test.py file. 

Your task is to write a new file that runs the inference on an entire video's sequence file
"""

GAUSSIAN_SIGMA = 2.0  # for smoothing


def load_model_from_checkpoint(
    checkpoint_path: str,
    input_size: int = 360,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    return_logits: bool = False,
):
    """Load model weights from checkpoint, adapting architecture if needed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and any(k.startswith('lstm.') or k.startswith('fc.') for k in ckpt.keys()):
        state_dict = ckpt
    else:
        # Fallback: attempt to use as state_dict
        state_dict = ckpt

    # Infer architecture from weights if possible
    inferred_input_size = input_size
    inferred_hidden_size = hidden_size
    inferred_num_layers = num_layers
    inferred_bidirectional = bidirectional

    try:
        # weight_ih_l0 shape: (4*hidden_size, input_size)
        w_ih_l0 = state_dict.get('lstm.weight_ih_l0', None)
        if w_ih_l0 is not None:
            inferred_hidden_size = w_ih_l0.shape[0] // 4
            inferred_input_size = w_ih_l0.shape[1]

        # Determine num_layers by counting layers
        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith('lstm.weight_ih_l'):
                try:
                    idx_str = k.split('lstm.weight_ih_l')[1]
                    idx = int(idx_str.split('_')[0]) if '_' in idx_str else int(idx_str)
                    layer_indices.add(idx)
                except Exception:
                    pass
        if layer_indices:
            inferred_num_layers = max(layer_indices) + 1

        # Bidirectionality: presence of any reverse weights
        inferred_bidirectional = any('_reverse' in k for k in state_dict.keys())
    except Exception:
        pass

    # Build model with inferred architecture
    model = TennisPointLSTM(
        input_size=inferred_input_size,
        hidden_size=inferred_hidden_size,
        num_layers=inferred_num_layers,
        dropout=0.2,
        bidirectional=inferred_bidirectional,
        return_logits=return_logits,
    )

    # Load strictly now that shapes should match
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(input_size={inferred_input_size}, hidden_size={inferred_hidden_size}, "
        f"num_layers={inferred_num_layers}, bidirectional={inferred_bidirectional})"
    )
    return model, device

# steps: 
# %%
# load model - best 300 sequence length model
model_path = 'checkpoints/seq_len300/best_model.pth'
model, device = load_model_from_checkpoint(model_path, bidirectional=True, return_logits=False)



# load whole feature npz file for a specific video
video_feature_path = 'pose_data/features/yolos_0.25conf_15fps_0s_to_99999s/Aditi Narayan ｜ Matchplay_features.npz'
feature_data = np.load(video_feature_path)

# create our ordered list of sequences with 50% overlap: must carefully track frame numbers

num_frames = len(feature_data['features'])
sequence_length = 300 
overlap = 150
if num_frames < sequence_length:
    raise ValueError("input video too short")


if num_frames % sequence_length == 0:
    # divides cleanly
    num_sequences = ((num_frames-sequence_length) // overlap) + 1
    start_idxs = [overlap*s for s in range(num_sequences)]

else:
    num_sequences_clean = ((num_frames-sequence_length) // overlap) + 1
    start_idxs = [overlap*s for s in range(num_sequences_clean)]
    start_idxs.append(num_frames - 1 - sequence_length) # adds last sequence

ordered_sequences = []
output_arr = np.full((3, num_frames), np.nan)


# now we construct the feature lists, perform inference, and fill output array, tracking start indexes
for i in start_idxs:
    feature_vec = feature_data['features'][i:i+sequence_length, :].shape
    output_sequence = model(feature_vec).detach().cpu().numpy()
    # now we do the nan checks:
    if output_arr[0, i:i+sequence_length].isnan().all(): # no overlap, can put in this row
        output_arr[0, i:i+sequence_length] = output_sequence
    elif output_arr[1, i:i+sequence_length].isnan().all():
        output_arr[1, i:i+sequence_length] = output_sequence
    elif i == start_idxs[-1] and num_frames % sequence_length > 0: # we allow 3x overlap for covering end sequence
        output_arr[2, i:i+sequence_length] = output_sequence
    else:
        raise ValueError('res arr filling logic messed up')

# now we have filled res_arr. next, get 1, num_frames array by averaging over 0th axis, and apply gaussian smoothing
avg_probs = np.nanmean(output_arr, axis=0)
smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs, sigma=GAUSSIAN_SIGMA)
# %%
smoothed_probs

# perform hysteresis filtering on smoothed sequence

# use hysteresis for start/end times, write to csv
