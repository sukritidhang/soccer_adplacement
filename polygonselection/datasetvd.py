# dataset.py
import os
import json
import cv2
import numpy as np
import math
from tqdm import tqdm

from generatemask_videodataset_allcoords import run_draw_all_coords
from generatemarked_videodataset_penalty import run_extract_penalty_coords
from generatepolygon_videodataset_p1p2p3 import run_polygon_p1p2p3
from choose_polygon_best_ver135 import run_polygon_best






def main():
    # === Define all paths here ===
    base_dir = os.path.expanduser('~/soccernet/soccernet_vd/dataset/Testing2')

    image_dir = os.path.join(base_dir, "frames-Testv2")
    json_folder = os.path.join(base_dir, "extremities-Testv2")
    #script3 outputs
    filtered_coords_dir = os.path.join(base_dir, "extractcoordsv2/filtered_coords-v2")
    mask_output_folder = os.path.join(base_dir, "extractcoordsv2/mask-v2")
    marked_output_folder = os.path.join(base_dir, "extractcoordsv2/marked-v2")

    # Script 2 outputs
    polygon_output_img_dir = os.path.join(base_dir, "polygon_marked-p1p2p3")
    polygon_output_json_dir = os.path.join(polygon_output_img_dir, "json_output")
    skipped_img_dir = os.path.join(polygon_output_img_dir, "skipped_images")
    skipped_json_dir = os.path.join(polygon_output_img_dir, "skipped_json")

     

    # Script 1 outputs
    best_output_img_dir = os.path.join(base_dir, "polygon-p1p2p3-best")
    best_output_json_dir = os.path.join(best_output_img_dir, "json_output")

    # Script 4 output
    mask_all_coords_dir = os.path.join(base_dir, "allcoords_masks")

    # Create necessary output folders
    for d in [
        mask_output_folder, marked_output_folder, filtered_coords_dir,
        polygon_output_img_dir, polygon_output_json_dir,
        best_output_img_dir, best_output_json_dir,
        mask_all_coords_dir, skipped_img_dir, skipped_json_dir 
    ]:
        os.makedirs(d, exist_ok=True)

    # === Run scripts with centralized paths ===
    print("🔁 Script 1: Draw All Polygons to Mask")
    run_draw_all_coords(image_dir, json_folder, mask_all_coords_dir)

    print("🔁 Script 2: Extract Coords penalty coords from Extremities")
    run_extract_penalty_coords(image_dir, json_folder, mask_output_folder,
                       marked_output_folder, filtered_coords_dir)

    print("🔁 Script 3: Generate P1, P2, P3 Polygons")
    run_polygon_p1p2p3(image_dir, filtered_coords_dir, polygon_output_img_dir,
                       polygon_output_json_dir, skipped_img_dir, skipped_json_dir )

    print("🔁 Script 4: Choose Best Polygon")
    run_polygon_best(image_dir, polygon_output_json_dir, best_output_img_dir,
                     best_output_json_dir)

    

    print("✅ All processing complete.")


if __name__ == "__main__":
    main()
