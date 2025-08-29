#!/usr/bin/env python3
"""
Test script to verify that the modified manual_court.py produces the same output as manual_court2.py
"""

import cv2
import numpy as np
import os
from manual_court import CourtDetector

def test_court_detection():
    """Test the court detection functionality"""
    
    # Initialize the court detector
    detector = CourtDetector()
    
    # Get list of video files
    video_dir = "raw_videos"
    if not os.path.exists(video_dir):
        print(f"Video directory {video_dir} not found")
        return
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    # Test with the first video
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Testing with video: {video_path}")
    
    try:
        # Process the video using the class-based approach
        out_mask, clean_frame, metadata = detector.process_video(video_path, target_time=60)
        
        print("Processing complete!")
        print(f"Metadata: {metadata}")
        
        # Save the results
        cv2.imwrite("test_out_mask.png", out_mask)
        cv2.imwrite("test_clean_frame.png", clean_frame)
        
        print("Results saved:")
        print("- test_out_mask.png (binary mask where white = out of bounds)")
        print("- test_clean_frame.png (clean frame without lines)")
        
        # Display the results
        print("Displaying results... Press any key to close")
        
        # Show the out mask
        cv2.namedWindow("Out Mask (White = Out of Bounds)", cv2.WINDOW_NORMAL)
        cv2.imshow("Out Mask (White = Out of Bounds)", out_mask)
        
        # Show the mask on the clean frame
        masked_frame = cv2.bitwise_and(clean_frame, clean_frame, mask=~out_mask) # invert the mask
        cv2.namedWindow("Masked Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Masked Frame", masked_frame)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_court_detection()
