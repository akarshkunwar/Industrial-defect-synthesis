import os
import cv2
import shutil
import yaml

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
MVTEC_ROOT = "./datasets"
SYNTHETIC_ROOT = "./synthetic_dataset"

# --- EXPERIMENTAL CONTROLS ---
# Set to False to build Baseline, True to build Augmented
INCLUDE_SYNTHETIC = True
# Output directory changes automatically to prevent accidental overwrites
YOLO_OUTPUT_DIR = "./yolo_dataset/yolo_metal_nut/augmented" if INCLUDE_SYNTHETIC else "./yolo_dataset/yolo_metal_nut/baseline"

CATEGORY = "metal_nut"
NUM_REAL_TRAIN = 5

# Dynamic Balancing: 20 for baseline, 'all' for augmented
NUM_GOOD_TRAIN = 'all' if INCLUDE_SYNTHETIC else 20

# The exact textural defect classes you generated data for.
TARGET_DEFECTS = ['color', 'scratch']


# ==========================================

def create_yolo_scaffolding(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[*] Clean scaffolding created at {base_dir}")


def generate_yolo_yaml(base_dir):
    yaml_path = os.path.join(base_dir, 'data.yaml')
    yaml_content = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['defect']
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"[*] Generated YOLO data.yaml at {yaml_path}")


def process_image_and_mask(img_path, mask_path, split, output_dir, prefix=""):
    img_name = os.path.basename(img_path)
    new_img_name = f"{prefix}_{img_name}" if prefix else img_name
    new_label_name = os.path.splitext(new_img_name)[0] + ".txt"

    dest_img_path = os.path.join(output_dir, 'images', split, new_img_name)
    dest_label_path = os.path.join(output_dir, 'labels', split, new_label_name)

    # Note: We now guarantee mask_path is valid before this function is even called
    # for synthetic data, ensuring we never copy an image without a label.
    if mask_path is not None and not os.path.exists(mask_path):
        print(f"[!] Critical: Mask path passed but does not exist: {mask_path}")
        return

    shutil.copy2(img_path, dest_img_path)

    if mask_path is None:
        return  # For background images

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[!] Warning: Could not read mask {mask_path}")
        return

    img_height, img_width = mask.shape
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(dest_label_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < 15:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + (w / 2.0)) / img_width
            y_center = (y + (h / 2.0)) / img_height
            norm_width = w / img_width
            norm_height = h / img_height

            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")


def ingest_mvtec_real_data(mvtec_dir, category, yolo_dir, num_train, num_good, target_defects):
    print(f"\n[*] Processing MVTec Real Data...")
    train_good_dir = os.path.join(mvtec_dir, category, 'train', 'good')
    test_good_dir = os.path.join(mvtec_dir, category, 'test', 'good')
    test_dir = os.path.join(mvtec_dir, category, 'test')
    gt_dir = os.path.join(mvtec_dir, category, 'ground_truth')

    if os.path.exists(train_good_dir):
        good_imgs = sorted([f for f in os.listdir(train_good_dir) if f.endswith(('.png', '.jpg'))])
        if num_good != 'all':
            good_imgs = good_imgs[:num_good]
        for img_file in good_imgs:
            process_image_and_mask(os.path.join(train_good_dir, img_file), None, 'train', yolo_dir, prefix="real_good")
        print(f"    -> train/good: {len(good_imgs)} images to Train")

    if os.path.exists(test_good_dir):
        test_good_imgs = sorted([f for f in os.listdir(test_good_dir) if f.endswith(('.png', '.jpg'))])
        for img_file in test_good_imgs:
            process_image_and_mask(os.path.join(test_good_dir, img_file), None, 'val', yolo_dir, prefix="real_good")
        print(f"    -> test/good: {len(test_good_imgs)} images to Val")

    for defect_type in os.listdir(test_dir):
        if defect_type == 'good' or defect_type not in target_defects:
            continue

        img_folder = os.path.join(test_dir, defect_type)
        if not os.path.isdir(img_folder):
            continue

        images = sorted([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))])
        train_imgs = images[:num_train]
        val_imgs = images[num_train:]

        for img_file in train_imgs:
            base_name = os.path.splitext(img_file)[0]
            mask_path = os.path.join(gt_dir, defect_type, f"{base_name}_mask.png")
            process_image_and_mask(os.path.join(img_folder, img_file), mask_path, 'train', yolo_dir,
                                   prefix=f"real_{defect_type}")

        for img_file in val_imgs:
            base_name = os.path.splitext(img_file)[0]
            mask_path = os.path.join(gt_dir, defect_type, f"{base_name}_mask.png")
            process_image_and_mask(os.path.join(img_folder, img_file), mask_path, 'val', yolo_dir,
                                   prefix=f"real_{defect_type}")

        print(f"    -> test/{defect_type}: {len(train_imgs)} train | {len(val_imgs)} val")


def ingest_synthetic_training(synthetic_root, category, yolo_dir, target_defects):
    print(f"\n[*] Ingesting Synthetic DoRA Data into Training ({category})...")
    category_dir = os.path.join(synthetic_root, category)

    if not os.path.exists(category_dir):
        print(f"[!] Synthetic category folder not found: {category_dir}")
        return

    for defect_type in os.listdir(category_dir):
        if defect_type not in target_defects:
            continue

        defect_dir = os.path.join(category_dir, defect_type)
        syn_images_dir = os.path.join(defect_dir, 'images')
        syn_masks_dir = os.path.join(defect_dir, 'masks')

        if not os.path.exists(syn_images_dir) or not os.path.exists(syn_masks_dir):
            continue

        count = 0
        mask_files_in_dir = os.listdir(syn_masks_dir)

        for img_file in os.listdir(syn_images_dir):
            if not img_file.endswith(('.png', '.jpg')):
                continue

            img_path = os.path.join(syn_images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            # THE FIX: Map 'cut_0199' to 'cut_mask_0199'
            if f"{defect_type}_" in base_name:
                expected_mask_base = base_name.replace(f"{defect_type}_", f"{defect_type}_mask_")
            else:
                expected_mask_base = f"{defect_type}_mask_{base_name}"

            mask_path = None
            for mf in mask_files_in_dir:
                if mf.startswith(expected_mask_base) and mf.endswith(('.png', '.jpg', '.tiff')):
                    mask_path = os.path.join(syn_masks_dir, mf)
                    break

            # If no mask is found, aggressively skip the image so we don't poison the dataset
            if mask_path is None:
                print(
                    f"[!] Warning: No matching mask found for {img_file} (Expected: {expected_mask_base}...). Skipping.")
                continue

            process_image_and_mask(img_path, mask_path, 'train', yolo_dir, prefix=f"synth_{defect_type}")
            count += 1

        print(f"    -> Synthetic {defect_type}: {count} labels generated successfully")

if __name__ == "__main__":
    mode_name = "Augmented" if INCLUDE_SYNTHETIC else "Baseline"
    print(f"--- Starting Filtered YOLO Dataset Compiler ({mode_name} Mode) ---")

    create_yolo_scaffolding(YOLO_OUTPUT_DIR)
    generate_yolo_yaml(YOLO_OUTPUT_DIR)

    ingest_mvtec_real_data(MVTEC_ROOT, CATEGORY, YOLO_OUTPUT_DIR, NUM_REAL_TRAIN, NUM_GOOD_TRAIN, TARGET_DEFECTS)

    if INCLUDE_SYNTHETIC:
        ingest_synthetic_training(SYNTHETIC_ROOT, CATEGORY, YOLO_OUTPUT_DIR, TARGET_DEFECTS)
    else:
        print("\n[*] Skipping Synthetic Data (Baseline Mode)")

    print("\n--- Process Finished! ---")
    print(f"Dataset compiled to: {YOLO_OUTPUT_DIR}")