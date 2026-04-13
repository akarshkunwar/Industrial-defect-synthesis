import os
import gc
import torch
import random
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

# Import your generation logic from the module we just cleaned up
from generate_defects import generate_synthetic_defect

# ==========================================
# CONFIGURATION
# ==========================================
GOOD_IMAGES_DIR = Path("C:/Users/akars/PycharmProjects/SD_LoRA_2.1/datasets/wood/train/good")
PREPARED_DATA_DIR = "./prepared_dora_data_wood"
OUTPUT_BASE_DIR = Path("./synthetic_dataset/wood")
TARGET_COUNT = 200 #no of defects to generate
DEFECT_TYPES = ["color"]#"color", "glue"]

# ADD THIS: Map each defect to its best checkpoint step.
# If a defect is mapped to None, it will default to the final weights.
"""CHECKPOINT_MAP = {
    "color": 250,  # 250-300
    "cut": 250,  # 350
    "fold": 350, # 100-200
    "glue": 250, #
    "poke": 450 # 100-200
}for leather"""
CHECKPOINT_MAP = {
    "color": None, # 100-250
    "hole": None, # 400
    "liquid": None # 400
} #for wood


def setup_directories():
    for defect in DEFECT_TYPES:
        (OUTPUT_BASE_DIR / defect / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE_DIR / defect / "masks").mkdir(parents=True, exist_ok=True)
    print(f"Created output directories at {OUTPUT_BASE_DIR.resolve()}")


def main():
    setup_directories()
    good_image_files = list(GOOD_IMAGES_DIR.glob("*.png"))

    if not good_image_files:
        raise FileNotFoundError(f"No good images found in {GOOD_IMAGES_DIR}")

    for defect in DEFECT_TYPES:
        print(f"\n{'=' * 50}\nStarting Batch Generation: {TARGET_COUNT} {defect.upper()} images\n{'=' * 50}")

        out_img_dir = OUTPUT_BASE_DIR / defect / "images"
        out_mask_dir = OUTPUT_BASE_DIR / defect / "masks"

        if defect == "flip":
            for i in tqdm(range(TARGET_COUNT), desc=f"Generating {defect}"):
                rand_img = random.choice(good_image_files)
                gen_img, mask_img = generate_synthetic_defect(defect, rand_img, PREPARED_DATA_DIR)
                gen_img.save(out_img_dir / f"{defect}_{i:04d}.png")
                mask_img.save(out_mask_dir / f"{defect}_mask_{i:04d}.png")
            continue

        # ==========================================
        # UPDATED: DYNAMIC CHECKPOINT LOADING
        # ==========================================
        target_step = CHECKPOINT_MAP.get(defect)

        if target_step is not None:
            # Route to the specific checkpoint folder you saved during training
            dora_path = f"./DoRA_Adapters/wood_dora/{defect}/checkpoints/step_{target_step}"
            print(f"Targeting specific checkpoint: Step {target_step}")
        else:
            # Fallback to final weights if no step is defined in the map
            dora_path = f"./DoRA_Adapters/wood_dora/{defect}/final_weights_{defect}"
            print("Targeting final weights (no specific checkpoint mapped).")

        if not os.path.exists(dora_path):
            print(f"Skipping {defect}: Adapter not found at '{dora_path}'.")
            continue

        print(f"Loading Base SD Model & Injecting '{defect}' DoRA...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        # Load the dynamically routed weights
        pipe.unet = PeftModel.from_pretrained(pipe.unet, dora_path)

        for i in tqdm(range(TARGET_COUNT), desc=f"Generating {defect}"):
            rand_img = random.choice(good_image_files)
            gen_img, mask_img = generate_synthetic_defect(defect, rand_img, PREPARED_DATA_DIR, class_name="wood",
                                                          pipe=pipe)
            gen_img.save(out_img_dir / f"{defect}_{i:04d}.png")
            mask_img.save(out_mask_dir / f"{defect}_mask_{i:04d}.png")

        print(f"Clearing VRAM after {defect} generation...")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print("\nBatch generation complete!")


if __name__ == "__main__":
    main()