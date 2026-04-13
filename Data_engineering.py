import os
import json
import shutil
from pathlib import Path


def prepare_specialist_datasets():
    # --- PATHS (Update these to your absolute paths!) ---
    mvtec_base = Path("C:/Users/akars/PycharmProjects/SD_LoRA_2.1/datasets/wood")

    # In MVTec, the actual real defects are stored in the 'test' folder,
    # while the 'ground_truth' folder holds their corresponding masks.
    defect_images_base = mvtec_base / "test"
    ground_truth_masks_base = mvtec_base / "ground_truth"

    # Where we will save the isolated training data
    output_base = Path("./prepared_dora_data_wood")

    defect_types = ["color", "hole", "liquid"]

    # The exact prompts the DoRA will associate with these visual features
    prompts = {
        "color": "a wood surface with a highly visible color mark defect, black or red stain, high contrast, discolouration",
        "hole": "a wood surface with a deep hole defect, drilled hole, physical cavity, realistic shadows inside the hole",
        "liquid": "a wood surface with a mud-brown liquid defect, droplets and smears, textured surface drop",
        "scratch": "a wood surface with a harsh scratch defect, deep grooves, or abrasion"
    }

    print("--- Starting Data Engineering for Specialist DoRAs ---")

    for defect in defect_types:
        print(f"\nProcessing {defect.upper()} dataset...")
        out_dir = output_base / defect
        out_dir.mkdir(parents=True, exist_ok=True)

        defect_images_dir = defect_images_base / defect
        defect_masks_dir = ground_truth_masks_base / defect

        if not defect_images_dir.exists() or not defect_masks_dir.exists():
            print(f"  [Warning] Missing MVTec folders for {defect}. Skipping.")
            continue

        jsonl_data = []
        valid_pairs = 0

        # Match each real defect image to its exact ground truth mask
        for img_path in defect_images_dir.glob("*.png"):
            # MVTec typically names masks like "000_mask.png" corresponding to image "000.png"
            mask_name = f"{img_path.stem}_mask.png"
            mask_path = defect_masks_dir / mask_name

            if not mask_path.exists():
                continue

            # Rename and copy to our isolated training folder
            new_img_name = f"image_{img_path.name}"
            new_mask_name = f"mask_{img_path.name}"

            shutil.copy(img_path, out_dir / new_img_name)
            shutil.copy(mask_path, out_dir / new_mask_name)

            # Create the exact metadata entry huggingface expects
            jsonl_data.append({
                "file_name": new_img_name,
                "mask_file_name": new_mask_name,
                "text": prompts[defect]
            })
            valid_pairs += 1

        # Write the metadata.jsonl file explicitly for this defect class
        with open(out_dir / "metadata.jsonl", "w") as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + "\n")

        print(f"  Success: Isolated {valid_pairs} training pairs for {defect}.")


if __name__ == "__main__":
    prepare_specialist_datasets()