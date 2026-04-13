import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
CATEGORY = "leather"  # Change to 'leather' or 'metal_nut'

# Where your original MVTec dataset lives (to grab the 'good' training images)
ORIGINAL_MVTEC_ROOT = "./datasets"

# Where your synthetic DoRA images and masks are currently saved
SYNTHETIC_ROOT = f"./synthetic_dataset/{CATEGORY}"

# Where the new Anomalib-ready dataset will be built
OUTPUT_DATASET_ROOT = "./patchcore_dataset"


# ==========================================

def create_anomalib_structure():
    print(f"--- Building PatchCore Dataset Structure for {CATEGORY} ---")

    out_category_dir = os.path.join(OUTPUT_DATASET_ROOT, CATEGORY)

    # Define target directories
    train_good_dir = os.path.join(out_category_dir, "train", "good")
    test_good_dir = os.path.join(out_category_dir, "test", "good")

    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)

    # 1. Copy the "Good" training images from the original MVTec folder
    print("[*] Copying 'good' training images...")
    orig_train_good = os.path.join(ORIGINAL_MVTEC_ROOT, CATEGORY, "train", "good")
    if os.path.exists(orig_train_good):
        for img in os.listdir(orig_train_good):
            shutil.copy(os.path.join(orig_train_good, img), os.path.join(train_good_dir, img))
    else:
        print(f"[!] Warning: Original 'good' train folder not found at {orig_train_good}")

    # 2. Copy the "Good" testing images from the original MVTec folder
    print("[*] Copying 'good' testing images...")
    orig_test_good = os.path.join(ORIGINAL_MVTEC_ROOT, CATEGORY, "test", "good")
    if os.path.exists(orig_test_good):
        for img in os.listdir(orig_test_good):
            shutil.copy(os.path.join(orig_test_good, img), os.path.join(test_good_dir, img))
    else:
        print(f"[!] Warning: Original 'good' test folder not found at {orig_test_good}")

    # 3. Process Synthetic Defects into the 'test' and 'ground_truth' folders
    print("[*] Migrating synthetic DoRA defects and masks...")

    for defect_type in os.listdir(SYNTHETIC_ROOT):
        defect_dir = os.path.join(SYNTHETIC_ROOT, defect_type)
        if not os.path.isdir(defect_dir):
            continue

        syn_images_dir = os.path.join(defect_dir, 'images')
        syn_masks_dir = os.path.join(defect_dir, 'masks')

        if not os.path.exists(syn_images_dir) or not os.path.exists(syn_masks_dir):
            continue

        # Create Anomalib test and ground_truth folders for this specific defect
        out_test_defect = os.path.join(out_category_dir, "test", f"synth_{defect_type}")
        out_gt_defect = os.path.join(out_category_dir, "ground_truth", f"synth_{defect_type}")

        os.makedirs(out_test_defect, exist_ok=True)
        os.makedirs(out_gt_defect, exist_ok=True)

        mask_files = os.listdir(syn_masks_dir)
        count = 0

        # Map images to masks just like we did for YOLO, but copy to the new structure
        for img_file in os.listdir(syn_images_dir):
            if not img_file.endswith(('.png', '.jpg')):
                continue

            base_name = os.path.splitext(img_file)[0]

            # Map 'cut_0199' to 'cut_mask_0199'
            if f"{defect_type}_" in base_name:
                expected_mask_base = base_name.replace(f"{defect_type}_", f"{defect_type}_mask_")
            else:
                expected_mask_base = f"{defect_type}_mask_{base_name}"

            mask_path = None
            for mf in mask_files:
                if mf.startswith(expected_mask_base) and mf.endswith(('.png', '.jpg', '.tiff')):
                    mask_path = os.path.join(syn_masks_dir, mf)
                    break

            if mask_path:
                # Copy Image to test/synth_defect/
                shutil.copy(os.path.join(syn_images_dir, img_file), os.path.join(out_test_defect, img_file))

                # Copy Mask to ground_truth/synth_defect/
                # Anomalib requires the mask to share the exact same string name as the image
                new_mask_name = img_file
                shutil.copy(mask_path, os.path.join(out_gt_defect, new_mask_name))

                count += 1

        print(f"    -> {defect_type}: Processed {count} Image/Mask pairs.")

    print(f"--- Formatting Complete! Dataset ready at {OUTPUT_DATASET_ROOT} ---")


if __name__ == "__main__":
    create_anomalib_structure()