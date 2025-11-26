import os
import cv2
import json
import numpy as np
from tqdm import tqdm

#image folder
#base_folder = "C:/Users/co33599/Documents/video_dataset_calib/Testing1"
#image_folder = os.path.join(base_folder, "frames-Testv1")
#json_folder = os.path.join(base_folder, "extremities-Testv1")
                           
#mask_output_folder = os.path.join(base_folder, "extractcoordsv1/mask-v1")
#marked_output_folder = os.path.join(base_folder, "extractcoordsv1/marked-v1")
#coords_output_folder = os.path.join(base_folder, "extractcoordsv1/filtered_coords-v1")


def run_extract_penalty_coords(image_dir, json_folder, mask_output_folder,
                       marked_output_folder, filtered_coords_dir):
    # Make sure output folders exist
    #os.makedirs(mask_output_folder, exist_ok=True)
    #os.makedirs(marked_output_folder, exist_ok=True)
    #os.makedirs(filtered_coords_dir, exist_ok=True)


    # === Keys to extract ===
    base_keys = [
        "Big rect.",
        "Big rect.",
        "Big rect.",
        "Small rect.",
        "Small rect.",
        "Small rect."
        "Side line"
    ]

    # === Get all image files ===
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

    print(f"Found {len(image_files)} images.")

    for image_name in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(image_dir, image_name)

        # Build corresponding JSON filename
        json_filename = "extremities_" + image_name.replace(".jpg", ".json")
        json_path = os.path.join(json_folder, json_filename)

        if not os.path.exists(json_path):
            print(f"[WARN] JSON not found for {image_name}, skipping.")
            continue

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image {image_name}")
            continue

        height, width = img.shape[:2]

        # Load JSON
        try:
            with open(json_path, "r") as f:
                coords = json.load(f)
        except Exception as e:
            print(f"[ERROR] Reading JSON for {image_name}: {e}")
            continue

        # Collect only required keys
        #desired_keys = base_keys + [k for k in coords if "Side line" in k and k not in base_keys]
        desired_keys = [k for k in coords.keys() if any(k.startswith(p) for p in base_keys)]
        filtered_coords = {k: coords[k] for k in desired_keys}

        #print(filtered_coords)

        mask = np.zeros((height, width), dtype=np.uint8)
        marked_img = img.copy()
        filtered_coords = {}

        for key in desired_keys:
            if key not in coords:
                continue

            try:
                pt1 = coords[key][0]
                pt2 = coords[key][1]

                x1 = int(pt1["x"] * width)
                y1 = int(pt1["y"] * height)
                x2 = int(pt2["x"] * width)
                y2 = int(pt2["y"] * height)

                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
                cv2.line(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                filtered_coords[key] = [pt1, pt2]
            except Exception as e:
                print(f"[ERROR] Drawing key {key} in {image_name}: {e}")

        # Save output files
        filename_base = os.path.splitext(image_name)[0]
        cv2.imwrite(os.path.join(mask_output_folder, f"{filename_base}_mask.png"), mask)
        cv2.imwrite(os.path.join(marked_output_folder, f"{filename_base}_marked.jpg"), marked_img)
        with open(os.path.join(filtered_coords_dir, f"{filename_base}_coords.json"), "w") as f_out:
            json.dump(filtered_coords, f_out, indent=4)

    print("✅ Batch processing complete.")

if __name__ == "__main__":
    run_extract_penalty_coords(image_dir, json_folder, mask_output_folder,
                       marked_output_folder, filtered_coords_dir)
