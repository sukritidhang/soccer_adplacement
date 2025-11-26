from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import scipy
import scipy.sparse
from scipy.sparse.linalg import spsolve
import json

from tqdm import tqdm

class BillboardIntegration:
    def __init__(self, source_path, target_path, device=None):
        #self.source = cv2.imread(source_path, cv2.IMREAD_COLOR)
        src = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

        if src.shape[2] == 4:  # has alpha
            bgr = src[:, :, :3]
            alpha = src[:, :, 3]

            # Keep as BGR + mask
            self.source = bgr
            self.alpha = alpha
        else:
            self.source = src
            self.alpha = np.ones(self.source.shape[:2], dtype=np.uint8) * 255
        self.target = cv2.imread(target_path, cv2.IMREAD_COLOR)
        

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def laplacian_matrix(self, n, m):
        mat_D = scipy.sparse.lil_matrix((m, m))
        mat_D.setdiag(-1, -1)
        mat_D.setdiag(-4)
        mat_D.setdiag(-1, 1)

        mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
        mat_A.setdiag(-1, 1 * m)
        mat_A.setdiag(-1, -1 * m)
        return mat_A

    # -----------------------------
    # LAB blending
    # -----------------------------

    def lab_blend(self, warped_ad, mask):
        target = self.target.copy()

        # Find bounding box of the mask
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            print("Warning: Mask is empty.")
            return target

        y_max, x_max = target.shape[:2]
        y_min, x_min = 0, 0

        # Region dimensions
        h = y_max - y_min
        w = x_max - x_min

        # Crop ROI
        source_crop = warped_ad[y_min:y_max, x_min:x_max]
        target_crop = target[y_min:y_max, x_min:x_max]
        mask_crop = mask[y_min:y_max, x_min:x_max]

        # Flatten dimensions
        mat_A = self.laplacian_matrix(h, w).tocsc()
        for y in range(1, h - 1):
          for x in range(1, w - 1):
              if mask[y, x] == 0:
                  k = x + y * w
                  mat_A[k, k] = 1
                  mat_A[k, k + 1] = 0
                  mat_A[k, k - 1] = 0
                  mat_A[k, k + w] = 0
                  mat_A[k, k - w] = 0
        mat_A = mat_A.tocsc()
        mask_flat = mask_crop.flatten()

        for channel in range(3):  # B, G, R
            src_flat = source_crop[:, :, channel].flatten()
            tgt_flat = target_crop[:, :, channel].flatten()

            alpha = 0.9
            mat_b = mat_A.dot(src_flat) * alpha
            mat_b[mask_flat == 0] = tgt_flat[mask_flat == 0]

            x = spsolve(mat_A, mat_b)
            x = np.clip(x.reshape((h, w)), 0, 255).astype('uint8')

            target[y_min:y_max, x_min:x_max, channel] = x

        return target


    def integrate_billboard(self, polygon_coords):#, save_fig_path=None
        target_h, target_w = self.target.shape[:2]

        # Convert polygon coords to pixel coordinates
        polygon_pts = np.array([[int(p['x']*target_w), int(p['y']*target_h)] for p in polygon_coords], dtype=np.int32)
        print('got polygon coords')

        # Polygon mask
        polygon_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon_pts], 255)
       
        # Resize ad to fit free space
        #ad_resized = cv2.resize(self.source, (w, h), interpolation=cv2.INTER_LINEAR)

        # Homography: source → free space rectangle
        pts_src = np.array([[0,0],[self.source.shape[1],0],[self.source.shape[1],self.source.shape[0]],[0,self.source.shape[0]]], dtype=np.float32)
        #ad placement without considering occlusion
        pts_dst2 = np.array([
            [int(p["x"] * target_w), int(p["y"] * target_h)]
            for p in polygon_coords
         ], dtype=np.float32)

        H2, status2 = cv2.findHomography(pts_src, pts_dst2)
        
        warped_ad2 = cv2.warpPerspective(self.source, H2, (target_w, target_h))
        warped_alpha1 = cv2.warpPerspective(self.alpha, H2, (target_w, target_h))

        combine_mask =  cv2.bitwise_and(polygon_mask, warped_alpha1)
        
        warped_ad_masked = cv2.bitwise_and(warped_ad2, warped_ad2, mask=combine_mask)

        targetbg = self.lab_blend(warped_ad_masked, combine_mask) #polygon_mask)
        
        '''
        warped_ad2 = cv2.warpPerspective(self.source, H2, (target_w, target_h))
        mask_3c = cv2.merge([polygon_mask, polygon_mask, polygon_mask])
        targetbg = self.lab_blend(warped_ad2, polygon_mask)
        #target_bgo = np.where(mask_3c > 0, warped_ad2, self.target)
        '''

        


        # Create a grid: 2 rows, 5 columns (adjust as needed)
        fig, axes = plt.subplots(1, 5, figsize=(20, 14))

        # Row 1
        axes[0].imshow(self.target[:,:,::-1])
        axes[0].set_title("Target Image"); axes[0].axis('off')

        axes[1].imshow(polygon_mask, cmap='gray')
        axes[1].set_title("Polygon mask"); axes[1].axis('off')

        axes[2].imshow(warped_ad2, cmap='grey')
        axes[2].set_title("warped_ad1")
        axes[2].axis('off')

        axes[3].imshow(targetbg[:,:,::-1])
        axes[3].set_title("Final Integrated Ad without occlusion"); axes[3].axis('off')
        
        axes[4].imshow(warped_ad_masked)
        axes[4].set_title("warped_ad_masked"); axes[4].axis('off')


        # Layout + save
        plt.tight_layout()
        #if save_fig_path:
            #fig.savefig(save_fig_path, dpi=300, bbox_inches='tight')

        #plt.show()
        plt.close(fig)


        return targetbg, fig

# --- Paths ---
#video_path = os.path.expanduser('~/soccernet/soccernet_vd/dataset/video/Testing1.mp4')
frames_dir = os.path.expanduser('C:/Users/co33599/Documents/video_dataset_calib/Testing1/frames-Testv1') 
polygon_json_dir = os.path.expanduser('C:/Users/co33599/Documents/video_dataset_calib/Testing1/polygon-p1p2p3-best/json_output') 
output_video_path = os.path.expanduser('C:/Users/co33599/Documents/video_dataset_calib/output/Testing1/outputvd_wooccl.mp4')
#save_fig_path = os.path.expanduser('~/soccernet/soccernet_vd/output/Testing1/visl')
 

# Ad image
source_path = os.path.expanduser('C:/Users/co33599/Documents/video_dataset_calib/newad/aut-removebg.png')

required_paths = [polygon_json_dir, frames_dir, source_path]#, save_fig_path
for path in required_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path not found: {path}")


# --- Prepare frames list ---
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
num_frames = len(frame_files)
print(frame_files[:10])
if num_frames == 0:
    raise ValueError(f"No frames found in {frames_dir}")

# Read first frame to get size

first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width = first_frame.shape[:2]

# FPS for output video
fps = 25

# Setup VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing {num_frames} frames...")

# Initialize your billboard class (update import if needed)
first_frame_path = os.path.join(frames_dir, frame_files[0])
billboard = BillboardIntegration(source_path, first_frame_path)

def reorder_polygon(points):
    pts = [(p['x'], p['y']) for p in points]
    # Sort by y (top to bottom)
    pts_sorted_by_y = sorted(pts, key=lambda p: p[1])

    # Bottom two points (larger y values)
    bottom_two = sorted(pts_sorted_by_y[2:], key=lambda p: p[0])
    bottom_left, bottom_right = bottom_two[0], bottom_two[1]

    # Top two points (smaller y values)
    top_two = sorted(pts_sorted_by_y[:2], key=lambda p: p[0])
    top_left, top_right = top_two[0], top_two[1]

    reordered = [bottom_left, top_left, top_right, bottom_right]
    print('reordered points')
    return [{'x': x, 'y': y} for x, y in reordered]

for frame_file in tqdm(frame_files, desc="Processing frames"):
    # Extract frame id, e.g. frame_00000.jpg -> 00000
    frame_id_str = frame_file.split("_")[1].split(".")[0]

    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)

    # Build polygon and mask paths according to your naming pattern
    json_path = os.path.join(polygon_json_dir, f"frame_{frame_id_str}_polygons.json")
    

    frame_processed = False

    if os.path.exists(json_path):
        with open(json_path) as jf:
            data = json.load(jf)

        polygon_coords = data.get("best_polygon", [])

        print(f"Frame {frame_id_str}: polygon points count = {len(polygon_coords)}")

        if len(polygon_coords) >= 4:
            polygon_coords = reorder_polygon(polygon_coords)
            

            try:
                billboard.target = frame.copy()
                #save_visualization = int(frame_id_str) % 20 == 0
                #if save_visualization
                   #fig_save_path = os.path.join(save_fig_path, f"frame_{frame_id_str}_visualization.png")

                result_frame, _ = billboard.integrate_billboard(polygon_coords)#, save_fig_path=fig_save_path)
                out.write(result_frame)
                frame_processed = True
            except Exception as e:
                print(f"Warning processing frame {frame_id_str}: {e}")

    if not frame_processed:
        out.write(frame)  # write original frame if no polygon or error


out.release()
print(f"Output video saved at {output_video_path}")
