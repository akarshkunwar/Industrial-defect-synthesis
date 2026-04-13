import os
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# Change this to 'wood', 'leather', or 'metal_nut' before running
CATEGORY = "metal_nut"

# Dynamic paths based on the chosen category
BASELINE_YAML = f'./yolo_dataset/yolo_{CATEGORY}/baseline/data.yaml'
AUGMENTED_YAML = f'./yolo_dataset/yolo_{CATEGORY}/augmented/data.yaml'
# ==========================================

def train_and_eval_yolo(yaml_path, run_name):
    print(f"\n{'=' * 50}\nStarting YOLOv8 Run: {run_name}\n{'=' * 50}")

    # Load YOLOv8 Nano
    model = YOLO('yolov8n.pt')

    # Train with Early Stopping (patience=30)
    model.train(
        data=yaml_path,
        epochs=300,
        imgsz=512,
        batch=16,
        workers=2,              # Reduced to prevent system RAM OOM
        cache=False,            # Prevents hoarding dataset in memory
        project='runs/detect',  # Forces output to the expected directory for plotting
        name=run_name,          # Dynamically named to prevent overwrites
        device=0,
        patience=30,
        verbose=False
    )

    # Validate against the hold-out test set
    metrics = model.val()
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    return map50, map50_95


if __name__ == '__main__':
    # 1. Run Baseline (5-Shot Real)
    base_run_name = f'Baseline_{CATEGORY}'
    base_map50, base_map95 = train_and_eval_yolo(
        BASELINE_YAML,
        base_run_name
    )

    # 2. Run Augmented (5 Real + 200 Synthetic)
    aug_run_name = f'Augmented_{CATEGORY}'
    aug_map50, aug_map95 = train_and_eval_yolo(
        AUGMENTED_YAML,
        aug_run_name
    )

    # 3. Print LaTeX-Ready Summary Table
    print("\n\n" + "=" * 60)
    print(f"FINAL YOLOv8 ABLATION RESULTS: {CATEGORY.upper()}")
    print("=" * 60)
    print(f"Model Run          | mAP@50  | mAP@50-95 | Improvement")
    print("-" * 60)
    print(f"Baseline (5-Shot)  | {base_map50:.4f}  | {base_map95:.4f}    | ---")
    print(f"Augmented (DoRA)   | {aug_map50:.4f}  | {aug_map95:.4f}    | +{(aug_map50 - base_map50):.4f}")
    print("=" * 60 + "\n")