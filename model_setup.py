import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


def setup_dora_model():
    print("--- Initializing SD Inpainting for 8GB VRAM ---")
    model_id = "runwayml/stable-diffusion-inpainting"

    # 1. Load the core components in fp16 to save massive amounts of VRAM
    weight_dtype = torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=weight_dtype, local_files_only=True)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype, local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=weight_dtype, local_files_only=True)

    # 2. Apply 8GB VRAM "Survival" Optimizations
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()

    # CRITICAL FIX 1: Universally compatible gradient tracking for frozen weights
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    unet.conv_in.register_forward_hook(make_inputs_require_grad)

    # 3. Configure DoRA (Weight-Decomposed Low-Rank Adaptation)
    print("--- Injecting DoRA Layers ---")
    dora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        use_dora=True,
        bias="none"
    )

    # 4. Wrap the UNet with the DoRA configuration
    unet = get_peft_model(unet, dora_config)

    # CRITICAL FIX 2: Cast ONLY the trainable DoRA parameters to float32 for math stability
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    # Move models to GPU
    unet.to("cuda")
    vae.to("cuda")
    text_encoder.to("cuda")

    # Print a summary of what is actually training
    unet.print_trainable_parameters()

    return tokenizer, text_encoder, vae, unet


if __name__ == "__main__":
    setup_dora_model()