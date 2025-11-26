import os
import json
import cv2
import numpy as np


import csv

def order_points_clockwise(pts):
    """Orders 4 points in clockwise order around their centroid"""
    pts = np.array(pts)
    center = np.mean(pts, axis=0)

    def angle_from_center(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])

    return pts[np.argsort([angle_from_center(p) for p in pts])]

# -------- Angle Calculation --------
def angle_between(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3 in degrees"""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0
    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# -------- Polygon Validation --------
def is_valid_polygon(polygon_coords, img_shape, angle_thresh=0):
    if not polygon_coords or len(polygon_coords) != 4:
        return False

    h, w = img_shape[:2]
    pts = np.array([[p["x"] * w, p["y"] * h] for p in polygon_coords], dtype=np.float32)
    pts = order_points_clockwise(pts)

    # Duplicate points check
    if len(np.unique(pts, axis=0)) < 4:
        return False

    # Area check
    area = cv2.contourArea(pts)
    if area < 35000: #100:
        return False

    # Convexity check
    if not cv2.isContourConvex(pts):
        return False

    # Aspect ratio check
    _, _, bw, bh = cv2.boundingRect(pts)
    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
    if aspect_ratio > 8:
        return False

    # Edge length check
    for i in range(4):
        d = np.linalg.norm(pts[i] - pts[(i + 1) % 4])
        if d < 20:
            return False

    # Angle checks
    angles = []
    for i in range(4):
        p1 = pts[i - 1]
        p2 = pts[i]
        p3 = pts[(i + 1) % 4]
        angle = angle_between(p1, p2, p3)
        angles.append(angle)

    min_angle = min(angles)
    max_angle = max(angles)
    if min_angle < angle_thresh or max_angle > (180 - angle_thresh):
        return False

    return True

# -------- Scoring --------
def compute_polygon_area(poly, img_shape):
    h, w = img_shape[:2]
    if len(poly) < 3:
        # Can't compute area for fewer than 3 points
        return 0.0
    pts = np.array([[p["x"] * w, p["y"] * h] for p in poly], dtype=np.float32)
    # Optionally reshape to (n,1,2) if needed by cv2.contourArea
    pts = pts.reshape((-1, 1, 2))
    return cv2.contourArea(pts)

def polygon_aspect_ratio(polygon_coords, img_shape):
    h, w = img_shape[:2]
    pts = np.array([[p["x"] * w, p["y"] * h] for p in polygon_coords], dtype=np.float32)
    
    if pts.size == 0 or pts.shape[0] < 3:
        return 0  # Invalid polygon
    
    rect = cv2.boundingRect(pts.astype(np.int32))  # Make sure it's int32
    bw, bh = rect[2], rect[3]
    
    if bh == 0:
        return 0
    return bw / (bh + 1e-5)

def choose_best_polygon(polygon_dict, img_shape):
    best_score = -float('inf')
    best_label = None
    best_polygon = None
    best_metrics = None

    print("\n--- Evaluating Polygons ---")
    for label, poly in polygon_dict.items():
        h, w = img_shape[:2]
        pts = np.array([[p["x"] * w, p["y"] * h] for p in poly], dtype=np.float32)
        pts = order_points_clockwise(pts)

        # Compute metrics
        aspect_ratio = polygon_aspect_ratio(poly, img_shape)
        angles = []
        for i in range(4):
            p1 = pts[i - 1]
            p2 = pts[i]
            p3 = pts[(i + 1) % 4]
            angles.append(angle_between(p1, p2, p3))
        area = compute_polygon_area(poly, img_shape)

        # Logging
        print(f"\n🔷 Polygon: {label}")
        print(f"  - Aspect Ratio: {aspect_ratio:.2f}")
        print(f"  - Angles: {[f'{a:.2f}' for a in angles]}")
        print(f"  - Area: {area:.2f}")

        if not is_valid_polygon(poly, img_shape):
            print("  ❌ Invalid polygon. Skipping.")
            continue

        if area > best_score:
            best_score = area
            best_polygon = poly
            best_label = label
            best_metrics = {
                "area": area,
                "aspect_ratio": aspect_ratio,
                "angles": angles
            }
            print("  ✅ Selected as current best.")

    if best_polygon is None:
        return None, None, None

    return best_label, best_polygon, best_metrics

# -------- Helpers --------
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# -------- Main Runner --------
def run_polygon_best(image_dir, polygon_output_json_dir, best_output_img_dir, best_output_json_dir):
    os.makedirs(best_output_img_dir, exist_ok=True)
    os.makedirs(best_output_json_dir, exist_ok=True)
    
    csv_path = os.path.join(best_output_img_dir, "polygon_metrics.csv")
    csv_header = [
        "image", "label", "area", "aspect_ratio",
        "angle1", "angle2", "angle3", "angle4",
        "is_valid", "is_selected"
    ]
    all_rows = []
    
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        json_name = base_name + "_polygons.json"
        json_path = os.path.join(polygon_output_json_dir, json_name)

        if not os.path.exists(json_path):
            print(f"JSON not found for {img_name}, skipping.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read {img_name}, skipping.")
            continue

        polygon_dict = load_json(json_path)
        if not polygon_dict:
            print(f"No polygons in {json_path}, skipping.")
            continue

        # Collect metrics for all polygons in this image
        polygons_metrics = []
        h, w = image.shape[:2]
        for label, poly in polygon_dict.items():
            pts = np.array([[p["x"] * w, p["y"] * h] for p in poly], dtype=np.float32)

            # Compute metrics
            aspect_ratio = polygon_aspect_ratio(poly, image.shape)
            angles = []
            for i in range(len(pts)):
                p1 = pts[i - 1]
                p2 = pts[i]
                p3 = pts[(i + 1) % len(pts)]
                angles.append(angle_between(p1, p2, p3))
            area = compute_polygon_area(poly, image.shape)
            valid = is_valid_polygon(poly, image.shape)

            polygons_metrics.append({
                "label": label,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "angles": angles,
                "is_valid": valid,
                "polygon": poly
            })

        # Choose best polygon (only valid ones)
        # create a dict filtered for valid polygons for choose_best_polygon
        valid_polygons = {m["label"]: m["polygon"] for m in polygons_metrics if m["is_valid"]}
        if not valid_polygons:
            print(f"No valid polygons for {img_name}, skipping.")
            continue

        best_label, best_polygon, _ = choose_best_polygon(valid_polygons, image.shape)
        if best_polygon is None:
            print(f"No valid 4-point polygon for {img_name}, skipping.")
            continue

        # Draw best polygon on image
        pts_best = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in best_polygon], np.int32)
        cv2.polylines(image, [pts_best], isClosed=True, color=(0, 255, 0), thickness=3)

        # Save image and best polygon JSON
        cv2.imwrite(os.path.join(best_output_img_dir, img_name), image)
        output_json = {
            "best_label": best_label,
            "best_polygon": best_polygon
        }
        with open(os.path.join(best_output_json_dir, json_name), 'w') as f:
            json.dump(output_json, f, indent=2)

        print(f"\n✅ Processed {img_name} -> Best polygon: {best_label}")

        # Add rows to CSV data with is_selected mark
        for m in polygons_metrics:
            all_rows.append([
                img_name,
                m["label"],
                m["area"],
                m["aspect_ratio"],
                *[f"{a:.2f}" for a in m["angles"]],
                "Yes" if m["is_valid"] else "No",
                "Yes" if m["label"] == best_label else "No"
            ])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(all_rows)

    print(f"\n📄 Saved polygon metrics CSV to: {csv_path}")

if __name__ == "__main__":
    run_polygon_best(image_dir, polygon_output_json_dir, best_output_img_dir,
                     best_output_json_dir)

