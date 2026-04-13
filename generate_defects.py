import random
import torch
import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


# ==========================================
# 1. CLASS-SPECIFIC MASK ROUTING
# ==========================================

def get_nut_geometry(image_bgr):
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if binary[0, 0] == 255:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((7, 7), np.uint8)
    clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    nut_metal_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    return nut_metal_mask


def get_valid_generation_zone_cv(image_bgr):
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_large)
    dilated_edges = cv2.dilate(closed_edges, kernel_large, iterations=2)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros((h, w), dtype=np.uint8)

    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) > (h * w * 0.01):
                cv2.drawContours(silhouette, [cnt], -1, 255, -1)

    bg_mask_sample = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(bg_mask_sample, (0, 0), (w, h), 255, 20)
    bg_pixels = gray[bg_mask_sample == 255]

    bg_median = np.median(bg_pixels) if len(bg_pixels) > 0 else 0
    bg_std = np.std(bg_pixels) if len(bg_pixels) > 0 else 0

    edge_protection = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
    tolerance = max(15, bg_std * 2)
    is_bg_color = np.abs(gray.astype(np.int16) - bg_median) < tolerance
    is_flat = (edge_protection == 0)

    true_holes = np.logical_and(is_bg_color, is_flat).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(silhouette, cv2.bitwise_not(true_holes))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    dt = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dt)

    if max_val == 0:
        return final_mask

    safe_zone = np.zeros_like(final_mask)
    safe_zone[dt > (max_val * 0.4)] = 255
    return safe_zone


def get_class_specific_mask(image_bgr, class_name):
    h, w = image_bgr.shape[:2]
    if class_name in ["wood", "leather", "carpet", "tile", "grid"]:
        return np.ones((h, w), dtype=np.uint8) * 255
    elif class_name == "metal_nut":
        return get_nut_geometry(image_bgr)
    else:
        return get_valid_generation_zone_cv(image_bgr)


# ==========================================
# 2. DEFECT PATCH PLACEMENT
# ==========================================

def get_spatially_aware_mask(defect_type, base_data_dir, valid_canvas_mask, class_name):
    data_path = Path(base_data_dir) / defect_type
    mask_files = list(data_path.glob("mask_*.png"))

    if not mask_files:
        raise FileNotFoundError(f"Could not find masks for '{defect_type}' in {data_path}")

    # 1. Load Mask
    real_mask_path = random.choice(mask_files)
    real_mask = cv2.imread(str(real_mask_path), cv2.IMREAD_GRAYSCALE)
    real_mask = cv2.resize(real_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    # 2. Dynamic Morphology (Protect thin defects!)
    if defect_type in ["cut", "fold", "color"]:
        kernel_clean = np.ones((2, 2), np.uint8)  # Gentle preservation
    else:
        kernel_clean = np.ones((5, 5), np.uint8)  # Standard blob cleanup

    real_mask = cv2.morphologyEx(real_mask, cv2.MORPH_CLOSE, kernel_clean)
    real_mask = cv2.morphologyEx(real_mask, cv2.MORPH_OPEN, kernel_clean)

    # 3. Contour Extraction (Fixes the disconnected component trap)
    if defect_type in ["hole", "liquid", "scratch"]:
        # 3. Universal Bounding Box (Captures ALL scattered holes/splatters)
        coords = cv2.findNonZero(real_mask)
        if coords is None:
            # If the mask is completely empty, return a blank mask
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("L")

        # This draws a single bounding box around EVERY white pixel in the mask
        x, y, w, h = cv2.boundingRect(coords)
        defect_patch = real_mask[y:y + h, x:x + w]
    else:
        contours, _ = cv2.findContours(real_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("L")

        # Grab the largest single defect shape in the mask
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        defect_patch = real_mask[y:y + h, x:x + w]

    # 4. Transformations
    max_dim = max(defect_patch.shape)
    if max_dim > 256:
        shrink_ratio = 256 / max_dim
        defect_patch = cv2.resize(defect_patch, (0, 0), fx=shrink_ratio, fy=shrink_ratio,
                                  interpolation=cv2.INTER_NEAREST)

    scale_factor = random.uniform(0.7, 1.4) if random.random() < 0.50 else 1.0
    new_w = int(defect_patch.shape[1] * scale_factor)
    new_h = int(defect_patch.shape[0] * scale_factor)
    if new_w > 0 and new_h > 0:
        defect_patch = cv2.resize(defect_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Rotation
    if random.random() < 0.95 and defect_type != "fold":
        angle = random.uniform(0, 360)
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        bound_w = int(new_h * abs_sin + new_w * abs_cos)
        bound_h = int(new_h * abs_cos + new_w * abs_sin)
        M[0, 2] += bound_w / 2 - center[0]
        M[1, 2] += bound_h / 2 - center[1]
        defect_patch = cv2.warpAffine(defect_patch, M, (bound_w, bound_h), flags=cv2.INTER_NEAREST)

    if random.random() < 0.50: defect_patch = cv2.flip(defect_patch, 1)
    if random.random() < 0.50: defect_patch = cv2.flip(defect_patch, 0)

    # Gentle random dilate/contract
    if random.random() < 0.70 and defect_type != "hole":
        morph_choice = random.choice(["dilate", "contract"])
        # Use a much safer dynamic kernel for randomized mutation
        mut_kernel = np.ones((3, 3), np.uint8) if defect_type not in ["cut", "fold", "color"] else np.ones((2, 2),
                                                                                                             np.uint8)

        if morph_choice == "dilate":
            defect_patch = cv2.dilate(defect_patch, mut_kernel, iterations=1)
        elif morph_choice == "contract":
            defect_patch = cv2.erode(defect_patch, mut_kernel, iterations=1)

        if cv2.countNonZero(defect_patch) == 0:
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("L")

    if defect_type == "hole" and random.random() < 0.70:
        mut_kernel = np.ones((3, 3), np.uint8)
        defect_patch = cv2.dilate(defect_patch, mut_kernel, iterations=1)

    # 5. Anchoring using Distance Transform Probability
    dt = cv2.distanceTransform(valid_canvas_mask, cv2.DIST_L2, 5)
    dt = dt.astype(np.float64)
    dt_sum = dt.sum()

    if dt_sum > 0:
        prob_map = dt / dt_sum
        prob_map_flat = prob_map.flatten()
        prob_map_flat /= prob_map_flat.sum()
        flat_idx = np.random.choice(dt.size, p=prob_map_flat)
        anchor_y, anchor_x = np.unravel_index(flat_idx, dt.shape)
    else:
        ys_valid, xs_valid = np.where(valid_canvas_mask > 0)
        if len(ys_valid) == 0:
            return Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("L")
        idx_canvas = random.randint(0, len(ys_valid) - 1)
        anchor_y, anchor_x = ys_valid[idx_canvas], xs_valid[idx_canvas]

    defect_ys, defect_xs = np.where(defect_patch > 0)
    idx_defect = random.randint(0, len(defect_ys) - 1)
    contact_y, contact_x = defect_ys[idx_defect], defect_xs[idx_defect]

    top_left_y = anchor_y - contact_y
    top_left_x = anchor_x - contact_x
    ph, pw = defect_patch.shape
    final_canvas = np.zeros_like(valid_canvas_mask)

    y1, y2 = max(0, top_left_y), min(512, top_left_y + ph)
    x1, x2 = max(0, top_left_x), min(512, top_left_x + pw)
    py1, py2 = max(0, -top_left_y), max(0, -top_left_y) + (y2 - y1)
    px1, px2 = max(0, -top_left_x), max(0, -top_left_x) + (x2 - x1)

    if y2 > y1 and x2 > x1:
        final_canvas[y1:y2, x1:x2] = defect_patch[py1:py2, px1:px2]

    final_mask = cv2.bitwise_and(final_canvas, valid_canvas_mask)

    # 6. Final Feathering
    #texture_classes = ["wood", "leather", "carpet", "tile", "grid"]
    #if defect_type == "color" and class_name in texture_classes:
    #    final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)
    #elif defect_type == "cut" or "fold" and class_name in texture_classes:
    #    final_mask = final_mask
    #else:
    #    _, final_mask = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)
    #    final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)

    return Image.fromarray(final_mask).convert("L")
# ==========================================
# 3. GENERATION FUNCTIONS
# ==========================================

def generate_opencv_flip(image_path, valid_canvas_mask):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (512, 512))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.add(h, 5)
    v = cv2.subtract(v, 15)
    modified_img = cv2.merge((h, s, v))
    modified_img = cv2.cvtColor(modified_img, cv2.COLOR_HSV2BGR)
    modified_img = cv2.GaussianBlur(modified_img, (3, 3), 0)
    modified_img = cv2.convertScaleAbs(modified_img, alpha=0.85, beta=10)

    mask_3c = cv2.cvtColor(valid_canvas_mask, cv2.COLOR_GRAY2BGR)
    composited_img = np.where(mask_3c == 255, modified_img, img)

    flip_axis = random.choice([0, 1])
    final_img = cv2.flip(composited_img, flip_axis)
    final_mask = cv2.flip(valid_canvas_mask, flip_axis)

    return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)), Image.fromarray(final_mask).convert("L")


def generate_synthetic_defect(defect_type, good_image_path, prepared_data_dir, class_name, pipe=None):
    img_bgr = cv2.imread(str(good_image_path))
    img_bgr = cv2.resize(img_bgr, (512, 512))

    valid_canvas_mask = get_class_specific_mask(img_bgr, class_name)

    if defect_type == "flip" and class_name == "metal_nut":
        return generate_opencv_flip(good_image_path, valid_canvas_mask)

    if pipe is None:
        raise ValueError(f"Pipeline required for generative defect '{defect_type}'.")

    base_image = Image.open(good_image_path).convert("RGB").resize((512, 512))

    # Pass class_name context for safe routing
    mask_image = get_spatially_aware_mask(defect_type, prepared_data_dir, valid_canvas_mask, class_name)

    if class_name == "leather":
        # Updated Prompt Dictionary for generate_defects.py
        prompts= {
            "color": "a highly contrastive, bright red chemical stain defect on a leather surface, deeply saturated, high visibility, bright red discoloration",
            "glue": "a thick, raised, highly translucent, glossy dried glue residue or droplet defect on leather, highly reflective and visible shiny surface, raised translucent blob",
            "fold": "a deep, structural fold defect in a leather surface, strong directional shadows, sharp 3D crease, warped physical texture, pinched material",
            "cut": "a sharp, deep cut defect in a leather surface, slashed material, dark inner shadow, exposed inner layer, sharp jagged edges",
            "poke": "a small, deep puncture poke defect in a leather surface, indented hole, structural damage, dark shadow at the center, pushed-in leather"
        }
    elif class_name == "metal_nut":
        prompts = {
            "scratch": f"a metal nut with a deep, highly visible scratch mark defect",
            "color": f"a metal nut with severe staining and discoloration defects",
        }
    elif class_name == "wood":
        prompts = {
            "color": f"a wood surface with a highly visible color mark defect, Red or Black stain, high contrast",
            "hole": f"a wood surface with a deep hole defect, drilled hole, or long physical ravine, realistic shadows inside the hole",
            "liquid": "a wood surface with a thick, mud-brown liquid spill defect, dirty brown fluid puddle, opaque muddy drop, dark brown water, catching the light",
            "scratch": f"a wood surface with a harsh scratch defect, visible surface abrasion, rough texture, harsh scuff marks"
        }
    else:
        prompts = {}

    prompt = prompts.get(defect_type, f"a {class_name.replace('_', ' ')} with a {defect_type} defect")
    negative_prompt = "blurry, low resolution, badly drawn, perfect condition, flawless, overexposed, random objects, background noise"

    # CRITICAL FIX 3: Dynamic Pipeline Tuning ONLY for textures
    #is_texture_color = (defect_type == "color" and class_name in ["wood", "leather", "carpet", "tile", "grid"])
    guidance = 12.0
    steps = 40

    generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 1000000))
    # ==========================================
    # NEW: FATTEN THE MASK FOR HOLE DEFECTS TO SURVIVE VAE COMPRESSION
    # ==========================================
    # Dilate the mask to give the UNet breathing room to draw shadows/depth
    gen_mask_np = np.array(mask_image)
    if defect_type in ["hole"]:
        # Fatten scattered defects significantly so they survive the VAE
        fat_kernel = np.ones((9, 9), np.uint8)
        gen_mask_np = cv2.dilate(gen_mask_np, fat_kernel, iterations=1)
        mask_image = Image.fromarray(gen_mask_np)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=mask_image,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]

    if  defect_type not in ["fold", "poke", "cut", "hole", "scratch", "color"]:
        #method commented out, but used specifically for leather color stains, and wood hole and liquid defects
        # to eliminate VAE bleed.
        # 1. Convert everything to numpy arrays
        base_np = np.array(base_image)
        gen_np = np.array(output)
        mask_np = np.array(mask_image)

        # 2. Force the mask to be strictly binary (0 or 255)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        mask_3c = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)

        # 3. Cookie-cutter the generated stain perfectly onto the background
        final_composited_np = np.where(mask_3c == 255, gen_np, base_np)

        # 4. Overwrite the output with the crisp composited image
        output = Image.fromarray(final_composited_np)
    if defect_type in ["color"] and class_name == "wood":
        # 1. Convert everything to numpy arrays
        base_np = np.array(base_image).astype(np.float32)
        gen_np = np.array(output).astype(np.float32)

        # 1. Binarize the mask to kill VAE bleed, then apply a TINY blur (micro-feather)
        _, mask_binary = cv2.threshold(np.array(mask_image), 127, 255, cv2.THRESH_BINARY)
        mask_micro_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)

        # 2. Normalize mask to 0.0 - 1.0 for alpha blending
        mask_alpha = cv2.cvtColor(mask_micro_blur, cv2.COLOR_GRAY2RGB) / 255.0

        # 3. Blend them mathematically (Stain sinks into the background)
        final_composited_np = (gen_np * mask_alpha) + (base_np * (1.0 - mask_alpha))

        output = Image.fromarray(final_composited_np.astype(np.uint8))

    return output, mask_image