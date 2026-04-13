import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update these paths to point to your actual YOLOv8 results.csv files
BASELINE_CSV = "./runs/detect/runs/detect/baseline_wood/results.csv"
AUGMENTED_CSV = "./runs/detect/runs/detect/augmented_wood/results.csv"

# Title for your graph (e.g., 'Metal Nut', 'Wood', 'Leather')
CATEGORY_NAME = "wood"


def clean_columns(df):
    """YOLOv8 CSVs have leading spaces in column names. This cleans them."""
    df.columns = df.columns.str.strip()
    return df


def plot_performance_curves():
    if not os.path.exists(BASELINE_CSV) or not os.path.exists(AUGMENTED_CSV):
        print("Error: Could not find one or both CSV files. Check your paths.")
        return

    # Load and clean data
    df_base = clean_columns(pd.read_csv(BASELINE_CSV))
    df_aug = clean_columns(pd.read_csv(AUGMENTED_CSV))

    # Create a figure with 2 subplots (side-by-side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Convergence and Performance - {CATEGORY_NAME}', fontsize=16, fontweight='bold')

    # --- Plot 1: Training Convergence (Validation Box Loss) ---
    # We use val/box_loss to show how well the model generalizes over epochs
    ax1.plot(df_base['epoch'], df_base['val/box_loss'], label='Baseline (5-Shot)', color='red', linestyle='--')
    ax1.plot(df_aug['epoch'], df_aug['val/box_loss'], label='Augmented (DoRA)', color='blue', linewidth=2)

    ax1.set_title('Validation Box Loss (Convergence)', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Box Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Plot 2: Performance Boundary (mAP@50-95) ---
    # mAP@50-95 is the strictest metric for localization precision
    ax2.plot(df_base['epoch'], df_base['metrics/mAP50-95(B)'], label='Baseline (5-Shot)', color='red', linestyle='--')
    ax2.plot(df_aug['epoch'], df_aug['metrics/mAP50-95(B)'], label='Augmented (DoRA)', color='blue', linewidth=2)

    ax2.set_title('Strict Localization Accuracy (mAP@50-95)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('mAP@50-95', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()

    output_filename = f"{CATEGORY_NAME.lower().replace(' ', '_')}_performance_curves.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully as: {output_filename}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    plot_performance_curves()