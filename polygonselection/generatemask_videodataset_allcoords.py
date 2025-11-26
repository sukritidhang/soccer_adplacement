import os
import json
import cv2
import numpy as np

# Paths
#image_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/frames-Testv1/"
#json_folder =  "C:/Users/co33599/Documents/video_dataset_calib/Testing1/extremities-Testv1/"
#mask_all_coords_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/allcoords_masks/"
##output_json_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/frames-Testv1/polygon_masks/json_output/"


def run_draw_all_coords(image_dir, json_folder, mask_all_coords_dir):
    #os.makedirs(output_mask_dir, exist_ok=True)
    #os.makedirs(output_json_dir, exist_ok=True)

    # Process each JSON file
    for json_file in os.listdir(json_folder):
        if not json_file.endswith(".json"):
            continue
        
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        base_name = os.path.splitext(json_file)[0].replace("extremities_", "")
        img_path = os.path.join(image_dir, base_name + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)  # single-channel mask
        output_json = {}

        # Find polygon keys automatically
        polygon_keys = [
            k for k in data.keys() 
            if isinstance(data[k], list) and len(data[k]) > 0 
            and 'x' in data[k][0] and 'y' in data[k][0]
        ]

        for i, key in enumerate(polygon_keys):
            pts = np.array([[int(p['x']*w), int(p['y']*h)] for p in data[key]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Fill polygon with label index (or just 255 if binary mask)
            #cv2.fillPoly(mask, [pts], 255, )
            cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=4)

            output_json[f"{key}_pixels"] = pts.squeeze().tolist()

        # Save mask image
        output_mask_path = os.path.join(mask_all_coords_dir, f"{base_name}_mask.png")
        cv2.imwrite(output_mask_path, mask)

        # Save pixel coordinates JSON
        #output_json_path = os.path.join(output_json_dir, f"{base_name}_pixels.json")
        #with open(output_json_path, "w") as f:
            #json.dump(output_json, f, indent=4)

        print(f"Processed {base_name}: polygons -> {polygon_keys}, mask saved at {output_mask_path}")


if __name__ == "__main__":
     run_draw_all_coords(image_dir, json_folder, mask_all_coords_dir)
