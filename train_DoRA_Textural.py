import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import gc
from torch.amp import autocast, GradScaler

# Import your working 8GB model setup!
from model_setup import setup_dora_model


class InpaintDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.entries = []

        with open(self.data_dir / "metadata.jsonl", "r") as f:
            for line in f:
                self.entries.append(json.loads(line))

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Outputs range [-1.0, 1.0]
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # 1. Load and process Image (RGB)
        img_path = self.data_dir / entry["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.img_transforms(image)

        # 2. Load and process Mask (Grayscale)
        mask_path = self.data_dir / entry["mask_file_name"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        # 3. Tokenize Text Prompt
        text_inputs = self.tokenizer(
            entry["text"], padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "input_ids": text_inputs.input_ids[0]
        }


import os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def run_validation(step, unet, vae, text_encoder, tokenizer, noise_scheduler, val_image, val_mask, prompt, out_dir):
    print(f"\n--- Running Validation at Step {step} ---")

    # Temporarily set UNet to evaluation mode
    unet.eval()

    # Create a temporary pipeline using the live models already in memory
    # We pass safety_checker=None to save precious VRAM
    pipeline = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    pipeline.set_progress_bar_config(disable=True)

    # Convert the normalized PyTorch tensors back to PIL Images for the pipeline
    # The image is currently [-1, 1], we need it in [0, 1] then to PIL
    val_image_pil = Image.fromarray(((val_image[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype("uint8"))
    val_mask_pil = Image.fromarray((val_mask[0, 0].cpu().numpy() * 255).astype("uint8"))

    # Generate the image
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            result = pipeline(
                prompt=prompt,
                image=val_image_pil,
                mask_image=val_mask_pil,
                num_inference_steps=40,  # Lower steps for a faster preview
                guidance_scale=7.5
            ).images[0]

    # Save the result
    os.makedirs(out_dir, exist_ok=True)
    result.save(os.path.join(out_dir, f"step_{step}.png"))

    # Clean up the pipeline wrapper and flip back to training mode
    del pipeline
    unet.train()
    print("--- Validation Complete. Resuming Training ---")

def train():
    defect_types = ["color"]
    base_data_dir = Path("./prepared_dora_data_wood")

    for defect in defect_types:
        dataset_dir = base_data_dir / defect

        if not dataset_dir.exists():
            print(f"Skipping {defect}: Directory {dataset_dir} not found.")
            continue

        print(f"\n{'=' * 50}")
        print(f"Starting DoRA Training for Specialist Adapter: {defect.upper()}")
        print(f"{'=' * 50}")

        print("--- Loading Base Model & Injecting DoRA ---")
        tokenizer, text_encoder, vae, unet = setup_dora_model()

        print("--- Loading Dataset ---")
        dataset = InpaintDataset(dataset_dir, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=2e-4)

        noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="scheduler", local_files_only=True)
        unet.train()

        # CRITICAL FIX 1: Initialize GradScaler for Mixed Precision
        scaler = GradScaler('cuda')

        step_mapping = {
            "color": 550,  # Simple texture change, metal nut, wood, leather
            "glue": 600,  # Additive surface blob
            #"scratch": 650,  # metal nut, minor structural damage with texture change
            "poke": 500,  # Minor structural damage
            "bent": 1000,  # Complex geometric distortion with lighting and shadows
            "fold": 500,  # Complex 3D geometric warp
            "cut": 500,  # Complex structural break with shadows
            "hole": 500, # Complex structural break with shadows, drill marks on edges
            "liquid": 600, # Additive surface blob with texture change, but no hard edges
            "scratch": 500 # wood, texture change only
        }

        # Grab the target steps, default to 1000 if a new defect isn't in the list
        target_steps = step_mapping.get(defect, 400)

        grad_accum_steps = 4
        dataset_size = len(dataloader)

        # Calculate epochs
        epochs = max(15, (target_steps * grad_accum_steps) // dataset_size)

        # Grab ONE fixed image/mask to use for consistent validation
        val_batch = next(iter(dataloader))
        val_prompt = dataset.entries[0]["text"]  # Grab the text prompt for the first image
        val_out_dir = f"./DoRA_Adapters/wood_dora/{defect}/validation_samples"

        ckpt_out_dir = f"./DoRA_Adapters/wood_dora/{defect}/checkpoints"
        os.makedirs(ckpt_out_dir, exist_ok=True)

        # Track actual optimizer steps (since we are doing gradient accumulation)
        global_step = 0

        print(f"Dataset size: {dataset_size} images.")
        print(f"--- Starting Training for {epochs} Epochs to hit target steps ---")

        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for step, batch in enumerate(progress_bar):
                images = batch["image"].to("cuda")
                masks = batch["mask"].to("cuda")
                input_ids = batch["input_ids"].to("cuda")

                masked_images = torch.where(masks < 0.5, images, torch.tensor(-1.0, device=images.device))

                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(input_ids)[0]
                        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                        masks_resized = F.interpolate(masks, size=(latents.shape[2], latents.shape[3]))
                        masked_image_latents = vae.encode(
                            masked_images).latent_dist.sample() * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    latent_model_input = torch.cat([noisy_latents, masks_resized, masked_image_latents], dim=1)
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                    loss = F.mse_loss(noise_pred, noise) / grad_accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1  # Increment our actual training step

                    # TRIGGER VALIDATION EVERY 100 STEPS
                    if global_step % 50 == 0:
                        run_validation(
                            step=global_step,
                            unet=unet,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            noise_scheduler=noise_scheduler,
                            val_image=val_batch["image"],
                            val_mask=val_batch["mask"],
                            prompt=val_prompt,
                            out_dir=val_out_dir
                        )

                        # 2. Save the DoRA weights
                        ckpt_path = os.path.join(ckpt_out_dir, f"step_{global_step}")

                        # Assuming you are using Hugging Face PEFT for the DoRA wrap:
                        unet.save_pretrained(ckpt_path)
                        print(f"--- Saved DoRA checkpoint to {ckpt_path} ---")

                epoch_loss += (loss.item() * grad_accum_steps)
                progress_bar.set_postfix({"loss": f"{(loss.item() * grad_accum_steps):.4f}"})

        output_dir = f"./DoRA_Adapters/wood_dora/{defect}/final_weights_{defect}"
        print(f"--- Training Complete! Saving Weights to {output_dir} ---")
        unet.save_pretrained(output_dir)

        print("--- Cleaning up VRAM for the next adapter ---")
        del tokenizer, text_encoder, vae, unet, optimizer, noise_scheduler, dataset, dataloader, scaler
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()