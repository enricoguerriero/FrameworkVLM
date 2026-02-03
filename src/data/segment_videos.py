import os
import cv2
import numpy as np

def export_clips(video_folder, annotation_folder, output_folder, clip_length, overlapping, frame_per_second=None):
    
    video_files = sorted(os.listdir(video_folder))
    annotation_files = sorted(os.listdir(annotation_folder))
    print(f"Found {len(video_files)} video files and {len(annotation_files)} annotation files.")
    
    for i, _ in enumerate(video_files):
        
        print("-" * 20)
        print(f"Processing video {i + 1}/{len(video_files)}")
        print("-" * 20)
        
        video_file = os.path.join(video_folder, video_files[i])
        annotation_file = os.path.join(annotation_folder, annotation_files[i])
        print(f"Processing {video_file} and {annotation_file}")
        
        annotation = read_annotations(annotation_file)
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
    
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video Original FPS: {original_fps}")

        if frame_per_second is None:
            target_fps = original_fps
            frame_interval = 1
            print("Target FPS: None (Keeping original FPS)")
        else:
            target_fps = frame_per_second
            frame_interval = max(1, int(original_fps / target_fps))
            print(f"Target FPS: {target_fps}")
            
        print(f"Frame interval: {frame_interval}")
        
        frame_per_clip = int(clip_length * target_fps)
        overlapping_frames = int(overlapping * frame_per_clip)
        
        print(f"Frames per clip: {frame_per_clip}")
        
        frame_index = 0
        frames_list = []
        clip_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
                
                if len(frames_list) == frame_per_clip:
                    current_time_ms = frame_index * (1000 / original_fps)
                    
                    active_labels = get_active_labels(
                        clip_start_ms=current_time_ms, 
                        clip_length_ms=clip_length * 1000, 
                        annotations=annotation
                    )
                    
                    if active_labels:
                        label_str = "_" + "_".join(active_labels)
                    else:
                        label_str = ""

                    file_name = f"video_{i}_clip_{clip_index}{label_str}.mp4"
                    output_path = os.path.join(output_folder, file_name)
                    height, width, layers = frames_list[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

                    for f in frames_list:
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out.release()

                    frames_list = frames_list[overlapping_frames:]
                    clip_index += 1
            
            frame_index += 1
            
        cap.release()
        print(f"Finished processing {video_file}")
        print(f"Exported {clip_index} clips from {video_file}")
        print("-" * 20)

def read_annotations(file_path):
    """
    Reads annotation file. No changes needed here usually, 
    but ensures it captures the full label string.
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            idx = next((i for i, p in enumerate(parts) if p.isdigit()), None)
            if idx is None or idx + 2 >= len(parts):
                continue
            
            label = ' '.join(parts[:idx])
            
            ann_start = int(parts[idx])
            ann_end = int(parts[idx + 1])
            ann_length = int(parts[idx + 2])
            annotations.append((label, ann_start, ann_end, ann_length))
    return annotations

def get_active_labels(clip_start_ms, clip_length_ms, annotations):
    """
    Returns a list of raw label strings that overlap >50% with the clip.
    """
    active_labels = []
    clip_end_ms = clip_start_ms + clip_length_ms
    threshold = clip_length_ms / 2

    for ann in annotations:
        ann_label, ann_start, ann_end, _ = ann
        
        start_overlap = max(clip_start_ms, ann_start)
        end_overlap = min(clip_end_ms, ann_end)
        dur = end_overlap - start_overlap
        
        if dur > threshold:
            if ann_label not in active_labels:
                active_labels.append(ann_label)
                
    return active_labels