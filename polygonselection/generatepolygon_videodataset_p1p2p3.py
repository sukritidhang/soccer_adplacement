import os
import json
import cv2
import numpy as np
import math

# Paths
#image_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/frames-Testv1/"
#filtered_coords_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/extractcoordsv1/filtered_coords-v1/"
#polygon_output_img_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/polygon_marked-p1p2p3/"
#polygon_output_json_dir = "C:/Users/co33599/Documents/video_dataset_calib/Testing1/polygon_marked-p1p2p3/json_output/"


def run_polygon_p1p2p3(image_dir, filtered_coords_dir, polygon_output_img_dir,
                       polygon_output_json_dir, skipped_img_dir, skipped_json_dir):

    #skipped_img_dir = os.path.join(output_img_dir, "skipped_images")
    #skipped_json_dir = os.path.join(output_img_dir, "skipped_json")

    #for d in [output_img_dir, output_json_dir, skipped_img_dir, skipped_json_dir]:
        #os.makedirs(d, exist_ok=True)

    # Colors for polygons: P1, P2, P3
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    def order_points_clockwise(points):
        """Order points in clockwise order around centroid."""
        if len(points) < 3:
            return points
        cx = sum(p['x'] for p in points) / len(points)
        cy = sum(p['y'] for p in points) / len(points)
        return sorted(points, key=lambda p: math.atan2(p['y'] - cy, p['x'] - cx))

    def draw_polygon(img, poly_coords, color, label=None, show_points=True):
        """Draw a closed polygon and return pixel coordinates."""
        h, w = img.shape[:2]
        if not poly_coords:
            return []
        pts = np.array([[int(p['x']*w), int(p['y']*h)] for p in poly_coords], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        for idx, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 5, color, -1)
            if show_points:
                text = f"c{idx}"  # Example: "c0"
                cv2.putText(img, text, (x + 5, y ), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
        if label:
            # Put label near centroid
            cx, cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
            cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return pts.tolist()

    def normalize_segment(seg, direction="ltl"):#direction="rtl"
        """
        Ensure segment is consistently ordered.
        direction = "rtl" (right-to-left) or "ltr" (left-to-right)
        """
        if len(seg) < 2:
            return seg
        p1, p2 = seg
        if direction == "ltl":
            return [p1, p2] if p1["x"] < p2["x"] else [p2, p1]
        if direction == "rtl":
            return [p1, p2] if p1["x"] > p2["x"] else [p2, p1]
        

    def assign_side_points(p_list, ref_point):
            """
            Given a 2-point list (like small_main), return the point that's closer to ref_point first.
            This ensures small_main[0] is near small_bottom[0], etc.
            """
            if len(p_list) != 2:
                return p_list
            d0 = (p_list[0]['x'] - ref_point['x'])**2 + (p_list[0]['y'] - ref_point['y'])**2
            d1 = (p_list[1]['x'] - ref_point['x'])**2 + (p_list[1]['y'] - ref_point['y'])**2
            return [p_list[0], p_list[1]] if d0 < d1 else [p_list[1], p_list[0]]
        
    def safe_assign(points, ref):
        if len(points) == 2 and ref:
            return assign_side_points(points, ref)
        return points

    # Process each JSON file
    for json_file in os.listdir(filtered_coords_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(filtered_coords_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        base_name = os.path.splitext(json_file)[0].replace("extremities_", "").replace("_coords", "")
        img_path = os.path.join(image_dir, base_name + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue

        debug_img = img.copy()
        
        
        if any(k.startswith("Small rect. left") or k.startswith("Big rect. left") for k in data):#if "Small rect. left top" in data:
            small_top = normalize_segment(data.get("Small rect. left top", []), "ltl")
            small_main = normalize_segment(data.get("Small rect. left main", []), "ltl")
            small_bottom = normalize_segment(data.get("Small rect. left bottom", []), "ltl")
            big_top = normalize_segment(data.get("Big rect. left top", []), "ltl")
            big_main = normalize_segment(data.get("Big rect. left main", []), "ltl")
            big_bottom = normalize_segment(data.get("Big rect. left bottom", []), "ltl")
        
        elif any(k.startswith("Small rect. right") or k.startswith("Big rect. right") for k in data):#elif "Small rect. right top" in data:
            small_top = normalize_segment(data.get("Small rect. right top", []), "rtl")
            small_main = normalize_segment(data.get("Small rect. right main", []), "rtl")
            small_bottom = normalize_segment(data.get("Small rect. right bottom", []), "rtl")
            big_top = normalize_segment(data.get("Big rect. right top", []), "rtl")
            big_main = normalize_segment(data.get("Big rect. right main", []), "rtl")
            big_bottom = normalize_segment(data.get("Big rect. right bottom", []), "rtl")
        else:
        # No coordinates found
          print(f"No left or right rects in {json_file}")
          continue


        if len(small_bottom) >= 1 and len(small_main) == 2:
            small_main = safe_assign(small_main, small_bottom[0])# if small_bottom else None) # assign_side_points(small_main, small_bottom[0])
        else:
            print(f"Skipping {json_file} due to missing small_bottom or small_main points.")
            #continue

        if len(big_bottom) >= 1 and len(big_main) == 2:
            big_main = safe_assign(big_main, big_bottom[0])# if big_bottom else None)
        else:
            print(f"Skipping {json_file} due to missing big_bottom or big_main points.")
            #continue

        if len(small_top) >= 1 and len(big_top) == 2:
            big_top = safe_assign(big_top, small_top[0])# if small_top else None)
        else:
            print(f"Skipping {json_file} due to missing small_top or big_top points.")

            #continue
        
        # Initialize empty polygons
        P1, P2, P3 = [], [], []

        
        if len(small_bottom) >= 1 and len(small_main) >= 1 and len(big_bottom) >= 1 and len(big_main) >= 1:
            if "left" in json_file:
                P1 = [big_bottom[0], small_bottom[0], small_main[0], big_main[0]]
                P1 = order_points_clockwise(P1)
            else:  # right side
                P1 = [big_bottom[0], small_bottom[0], small_main[0], big_main[0]]
                P1 = order_points_clockwise(P1)

        # --- Build P2 (middle quadrilateral) ---
        if len(small_main) == 2 and len(big_main) == 2:
            P2 = [small_main[0], small_main[1], big_main[1], big_main[0]]
            P2 = order_points_clockwise(P2)

        # --- Build P3 explicitly ---
        if len(small_top) >= 1 and len(small_main) == 2 and len(big_top) == 2:
            chosen_small_top = small_top[0]

            d0 = (small_main[0]['x'] - chosen_small_top['x'])**2 + (small_main[0]['y'] - chosen_small_top['y'])**2
            d1 = (small_main[1]['x'] - chosen_small_top['x'])**2 + (small_main[1]['y'] - chosen_small_top['y'])**2
            chosen_small_main = small_main[0] if d0 < d1 else small_main[1]
            if "left" in json_file:
                P3 = [small_top[0], chosen_small_main, big_main[1], big_top[0]]
                P3 = order_points_clockwise(P3)
            else:  # right side
                P3 = [small_top[0], chosen_small_main, big_top[0], big_top[1]]
                P3 = order_points_clockwise(P3)
            


        if P1:
            P1_pixels = draw_polygon(debug_img, P1, colors[0], "P1")
        if P2:
            P2_pixels = draw_polygon(debug_img, P2, colors[1], "P2")
        if P3:
            P3_pixels = draw_polygon(debug_img, P3, colors[2], "P3")

        if not P1 and not P2 and not P3:
            # all polygons missing → treat as skipped
            img_out = os.path.join(skipped_img_dir, f"{base_name}_skipped.jpg")
            json_out = os.path.join(skipped_json_dir, f"{base_name}_skipped.json")
        else:
            img_out = os.path.join(polygon_output_img_dir, f"{base_name}_marked.jpg")
            json_out = os.path.join(polygon_output_json_dir, f"{base_name}_polygons.json")

        cv2.imwrite(img_out, debug_img)
        with open(json_out, "w") as f:
            json.dump({"P1": P1, "P2": P2, "P3": P3}, f, indent=4)

        print(f"Processed {base_name}: saved to {'SKIPPED' if not P1 and not P2 and not P3 else 'NORMAL'}")
        
        '''
        # Save marked image
        output_img_path = os.path.join(output_img_dir, f"{base_name}_marked.jpg")
        cv2.imwrite(output_img_path, debug_img)

        # Save JSON with polygons and pixel coordinates
        output_json = {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            #"P1_pixels": P1_pixels,
            #"P2_pixels": P2_pixels,
            #"P3_pixels": P3_pixels
        }
        output_json_path = os.path.join(output_json_dir, f"{base_name}_polygons.json")
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

        print(f"Processed {base_name}: P1, P2, P3 drawn and saved")
        '''

if __name__ == "__main__":

    run_polygon_p1p2p3(image_dir, filtered_coords_dir, polygon_output_img_dir,
                       polygon_output_json_dir, skipped_img_dir, skipped_json_dir)
