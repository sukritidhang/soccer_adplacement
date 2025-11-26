import cv2
import numpy as np
import matplotlib.pyplot as plt

def framewise_flicker(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diffs = []

    # Skip frames until 'start'
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, prev = cap.read()
    if not ret:
        print("Couldn't read starting frame.")
        return []
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for i in range(start+1, min(end+1, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
        frame_diffs.append(diff)
        prev_gray = gray

    cap.release()
    return frame_diffs





start=0
end=267
flick_org = framewise_flicker("C:/Users/co33599/Documents/video_dataset_calib/Football Testing/Testing3.mp4", start, end)
flick_mod = framewise_flicker("C:/Users/co33599/Documents/video_dataset_calib/output/Testing3/output_video.mp4", start, end)

frames = list(range(start, end))
plt.figure(figsize=(10,4))
plt.plot(frames, flick_org, label="Original", linestyle="--")
plt.plot(frames, flick_mod, label="Advert-inserted", linewidth=2)
plt.xlabel("Frames")
plt.ylabel("Flicker Index")
plt.xticks(np.arange(0,267 , 20)) #2
plt.title("Frame-wise flicker index (Frames 0–267)")
plt.legend()
#plt.grid(True)
plt.show()


def framewise_flow_consistency(video1, video2, start, end):
    """
    Computes per-frame optical flow difference between two videos (original and modified).
    Returns a list of flow differences for each frame in [start, end].
    """
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Jump to the starting frame
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    ret1, prev1 = cap1.read()
    ret2, prev2 = cap2.read()
    if not (ret1 and ret2):
        print("Error: could not read starting frames.")
        return []
    
    prev1_gray = cv2.cvtColor(prev1, cv2.COLOR_BGR2GRAY)
    prev2_gray = cv2.cvtColor(prev2, cv2.COLOR_BGR2GRAY)
    
    flow_diffs = []

    for i in range(start + 1, min(end+1, total_frames)):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow for both videos
        flow1 = cv2.calcOpticalFlowFarneback(prev1_gray, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(prev2_gray, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Compute per-frame flow difference (L1 norm)
        diff = np.mean(np.abs(flow1 - flow2))
        flow_diffs.append(diff)
        
        prev1_gray, prev2_gray = gray1, gray2
    
    cap1.release()
    cap2.release()
    return flow_diffs

flick_org =("C:/Users/co33599/Documents/video_dataset_calib/Football Testing/Testing1.mp4")
flick_mod = ("C:/Users/co33599/Documents/video_dataset_calib/output/Testing1/output_video.mp4")

start=0
end=267
flow_diffs = framewise_flow_consistency(flick_org, flick_mod, start, end)
 

# Frame indices for plotting
#frames = list(range(221, 240 + len(flow_diffs) + 1))
frames = list(range(start, end))
# Plot
plt.figure(figsize=(10, 4))
plt.plot(frames, flow_diffs, color='blue', linewidth=2)
plt.xlabel("Frames")
plt.ylabel("Optical Flow Difference")
plt.title("Frame-wise optical-flow consistency (Frames 0–267)")
plt.xticks(np.arange(0, 267, 20))
#plt.grid(True)
plt.tight_layout()
plt.show()


