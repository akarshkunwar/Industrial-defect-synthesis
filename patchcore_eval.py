import os
from anomalib import TaskType
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import MVTecAD
from lightning.pytorch import seed_everything
from huggingface_hub import login

# ==========================================
# AUTHENTICATION
# ==========================================
HF_TOKEN = ""
login(token=HF_TOKEN)


def run_patchcore(dataset_path, category_name, run_name):
    print(f"\n--- Starting PatchCore Evaluation for {run_name} ({category_name}) ---")

    # 1. Setup the MVTecAD DataModule
    # Removed 'task' here to fix the MVTecAD.__init__ error
    datamodule = MVTecAD(
        root=dataset_path,
        category=category_name,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=2
    )

    # 2. Initialize PatchCore
    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
        coreset_sampling_ratio=0.1
    )
    # MANUALLY SET THE TASK: This avoids the constructor error
    model.task = TaskType.SEGMENTATION
    # 3. Setup the Engine
    # Removed 'task' here to fix the Trainer.__init__ error
    engine = Engine(
        default_root_dir=f"./patchcore_results/{run_name}",
        accelerator="gpu",
        devices=1
    )

    # 4. Train
    print("Extracting coreset features...")
    engine.fit(datamodule=datamodule, model=model)

    # 5. Evaluate
    print("Evaluating AUROC boundaries...")
    test_results = engine.test(datamodule=datamodule, model=model)

    # Extract results
    print("\nRaw Results Dict:", test_results)

    # Fallback logic for keys because versioning changes these too
    image_auroc = 0.0
    pixel_auroc = 0.0
    if test_results:
        res = test_results[0]
        image_auroc = res.get('image_AUROC', res.get('test_image_AUROC', res.get('Metrics/image_AUROC', 0.0)))
        pixel_auroc = res.get('pixel_AUROC', res.get('test_pixel_AUROC', res.get('Metrics/pixel_AUROC', 0.0)))

    print("\n==========================================")
    print(f"RESULTS FOR: {run_name} - {category_name}")
    print(f"Image AUROC: {image_auroc:.5f}")
    print(f"Pixel AUROC: {pixel_auroc:.5f}")
    print("==========================================\n")


if __name__ == '__main__':
    seed_everything(42, workers=True)

    CURRENT_CATEGORY = "leather"

    # Baseline Test
    run_patchcore(
        dataset_path="./datasets",
        category_name=CURRENT_CATEGORY,
        run_name=f"{CURRENT_CATEGORY.capitalize()}_Baseline"
    )

    # Augmented Test
    run_patchcore(
        dataset_path="./patchcore_dataset",
        category_name=CURRENT_CATEGORY,
        run_name=f"{CURRENT_CATEGORY.capitalize()}_Augmented"
    )