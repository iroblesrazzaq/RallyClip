# file to run inference on an entire feature npz file, then apply postprocessing steps
# to output final start_time,end_time csv file
import numpy as np
"""
My test.py file is my current evaluation file. however, it just looks at sequences, not the whole video. 
I need to further establish my post processing pipeline. The final output of my pipeline should be 
a csv of start_time, end_times, which i can compare to the annotated targets. 
For now, we will use the same gaussian smoothing and hysteresis filtering that we're using in the test.py file. 

Your task is to write a new file that runs the inference on an entire video's sequence file
"""

# steps: 

# load model - best 300 sequence length model
model_path = 'checkpoints/seq_len300/best_model.pth'


# load whole feature npz file for a specific video
video_feature_path = 'pose_data/features/Monica Greene unedited tennis match play.npz'
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
    start_idxs = [150*s for s in range(num_sequences)]
else:
    num_sequences_clean = ((num_frames-sequence_length) // overlap) + 1
    start_idxs = [150*s for s in range(num_sequences_clean)]
    start_idxs.append(num_frames - 1 - sequence_length) # adds last sequence

ordered_sequences = []
res_arr = np.full((2, num_frames), np.nan)


# now we construct the feature lists, tracking start indexes


# just create num frames x2 array, fill with nan. then to check whether to put first or second row,
# just check if first row is nan, if so then fill first row, if already filled, then put into row 2
# ok, so if randomly filling in b/c dict key vals, then we can just check .isnan.any() on the sequence of note.



'''
case 700 (if not divisible by sequence length)

0-300
150-450
300-600
400-700 




case 600 (if divisible by sequence length)
0-300
150-450
300-600

= num_frames-sequence_length) // overlap + 1




'''


# perform inference on each individual sequence


# now create final output sequence by merging all sequences, averaging all overlapping frame predictions. 

# perform gaussian smoothing on that probability sequence

# perform hysteresis filtering on smoothed sequence

# use hysteresis for start/end times, write to csv
