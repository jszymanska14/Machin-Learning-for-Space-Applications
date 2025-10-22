import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(path):
    img = Image.open(path)
    return np.array(img)


def calculate_histogram(image_data, channel):
    if len(image_data.shape) == 2:
        channel_data = image_data.flatten()
    else:
        channel_data = image_data[:, :, channel].flatten()
    hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
    return hist, bins


def plot_histogram_rgb(image_data, title, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist, bins = calculate_histogram(image_data, i)
        axes[i].bar(bins[:-1], hist, width=1, color=color, alpha=0.7)
        axes[i].set_title(f'{title} - {name} Channel')
        axes[i].set_xlabel('Pixel Intensity')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_histogram_sar(vv_data, vh_data, title, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    hist_vv, bins_vv = np.histogram(vv_data.flatten(), bins=256, range=(0, 256))
    axes[0].bar(bins_vv[:-1], hist_vv, width=1, color='blue', alpha=0.7)
    axes[0].set_title(f'{title} - VV Polarization')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    hist_vh, bins_vh = np.histogram(vh_data.flatten(), bins=256, range=(0, 256))
    axes[1].bar(bins_vh[:-1], hist_vh, width=1, color='green', alpha=0.7)
    axes[1].set_title(f'{title} - VH Polarization')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def replace_2_lsb_random(image_data):
    manipulated = image_data.copy()

    for i in range(manipulated.shape[0]):
        for j in range(manipulated.shape[1]):
            for k in range(3):
                pixel = manipulated[i, j, k]
                cleared = pixel & 0b11111100
                random_bits = np.random.randint(0, 4)
                manipulated[i, j, k] = cleared | random_bits

    return manipulated


def check_visual_quality(original, manipulated, image_name):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original)
    axes[0].set_title(f'Original Image - {image_name}')
    axes[0].axis('off')

    axes[1].imshow(manipulated)
    axes[1].set_title(f'Manipulated Image - {image_name}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    diff = np.abs(original.astype(float) - manipulated.astype(float))
    print(f"\n{image_name} - Visual Quality Analysis:")
    print(f"  Maximum pixel difference: {diff.max()}")
    print(f"  Average pixel difference: {diff.mean():.4f}")
    print(f"  Visual quality impact: Minimal (max change of 3/255 = 1.18%)")


def compare_histograms(original, manipulated, image_name):

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']

    fig.suptitle(f'Histogram Comparison - {image_name}', fontsize=14, fontweight='bold')

    for i, (color, name) in enumerate(zip(colors, channel_names)):
        hist_orig, bins = calculate_histogram(original, i)
        axes[i, 0].bar(bins[:-1], hist_orig, width=1, color=color, alpha=0.7)
        axes[i, 0].set_title(f'Original - {name} Channel')
        axes[i, 0].set_xlabel('Pixel Intensity')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].grid(True, alpha=0.3)

        hist_manip, bins = calculate_histogram(manipulated, i)
        axes[i, 1].bar(bins[:-1], hist_manip, width=1, color=color, alpha=0.7)
        axes[i, 1].set_title(f'Manipulated - {name} Channel')
        axes[i, 1].set_xlabel('Pixel Intensity')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def detect_manipulation_analysis(original, manipulated, image_name):
    print(f"\n=== Manipulation Detection Analysis - {image_name} ===\n")

    for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
        hist_orig, _ = calculate_histogram(original, i)
        hist_manip, _ = calculate_histogram(manipulated, i)

        chi_square = np.sum((hist_orig - hist_manip) ** 2 / (hist_manip + 1e-10))

        print(f"{channel_name} Channel:")
        print(f"  Chi-square statistic: {chi_square:.2f}")
        print(f"  Histogram difference sum: {np.sum(np.abs(hist_orig - hist_manip))}")

    print("\nConclusion:")
    print("Yes, manipulation can be detected through:")
    print("  - Chi-square test on histogram distributions")
    print("  - LSB uniformity analysis")
    print("  - Sample Pairs Analysis (SPA)")
    print("  - RS Steganalysis")
    print("  - Statistical anomalies in pixel value distributions")


if __name__ == "__main__":
    print("=== SENTINEL DATA ANALYSIS & STEGANOGRAPHY ===\n")

    # 1. PDF of Sentinel Data
    natural_area_s2 = load_image('/Users/juliaszymanska/MachineLearning/natura2.jpg')
    urban_area_s2 = load_image('/Users/juliaszymanska/MachineLearning/Warsaw_Poland.jpg')

    print("=" * 70)
    print("TASK 1: PDF of Sentinel Data")
    print("=" * 70)


    print("\n1.1 Accessing Sentinel 2 data (natural and urban areas)")


    print("1.2 Drawing histogram of images\n")

    print("Natural Area (Sentinel 2):")
    plot_histogram_rgb(natural_area_s2, "Sentinel 2 - Natural Area")

    print("\nUrban Area - Warsaw (Sentinel 2):")
    plot_histogram_rgb(urban_area_s2, "Sentinel 2 - Urban Area (Warsaw)")


    print("\n1.3 Observations (Sentinel 2 - Optical Data):")
    print("\nNatural Area:")
    print("  - Higher green channel values (vegetation)")
    print("  - Red channel shows chlorophyll absorption")
    print("  - Blue channel influenced by atmospheric scattering")
    print("  - Clear bimodal distribution in green channel (vegetation vs. non-vegetation)")
    print("\nUrban Area:")
    print("  - More balanced distribution across RGB channels")
    print("  - Higher intensity values (concrete, asphalt reflection)")
    print("  - Less variation in green channel compared to natural areas")
    print("  - More uniform distribution due to man-made structures")


    print("\n" + "=" * 70)
    print("1.4 Repeat for Sentinel 1 data from the same regions")
    print("=" * 70)

    try:
        natural_vv_s1 = load_image('/Users/juliaszymanska/MachineLearning/natura_s1_vv.jpg')
        natural_vh_s1 = load_image('/Users/juliaszymanska/MachineLearning/natura_s1_vh.jpg')
        urban_vv_s1 = load_image('/Users/juliaszymanska/MachineLearning/warsaw_s1_vv.jpg')
        urban_vh_s1 = load_image('/Users/juliaszymanska/MachineLearning/warsaw_s1_vh.jpg')

        print("\nNatural Area (Sentinel 1 - SAR):")
        plot_histogram_sar(natural_vv_s1, natural_vh_s1, "Sentinel 1 - Natural Area")

        print("\nUrban Area - Warsaw (Sentinel 1 - SAR):")
        plot_histogram_sar(urban_vv_s1, urban_vh_s1, "Sentinel 1 - Urban Area (Warsaw)")

        print("\n1.4 Observations (Sentinel 1 - SAR Data):")
        print("\nNatural Area:")
        print("  - VV polarization: Lower backscatter from vegetation (volume scattering)")
        print("  - VH polarization: Shows forest structure and vegetation density")
        print("  - Cross-polarization (VH) more sensitive to vegetation structure")
        print("\nUrban Area:")
        print("  - VV polarization: High backscatter from buildings (double-bounce)")
        print("  - Strong returns from vertical structures and corners")
        print("  - Clear distinction between built-up areas and open spaces")
        print("  - Higher intensity values compared to natural areas")

    except FileNotFoundError:
        print("\n[INFO] Sentinel 1 data not found at specified paths.")
        print("To complete this part, you need to:")
        print("  1. Download Sentinel 1 SAR data (VV and VH polarizations)")
        print("  2. From the SAME geographic regions as your Sentinel 2 data")
        print("  3. Update the file paths in the code")
        print("  4. Sentinel 1 data source: https://scihub.copernicus.eu/")

    # 2. Steganography
    print("\n" + "=" * 70)
    print("TASK 2: Steganography")
    print("=" * 70)
    print("\nUsing images from Task 1 (Sentinel 2 data)\n")

    images_to_process = [
        (natural_area_s2, "Natural Area", "manipulated_natura2.png"),
        (urban_area_s2, "Urban Area (Warsaw)", "manipulated_warsaw.png")
    ]

    for idx, (original_image, image_name, save_filename) in enumerate(images_to_process, 1):
        print(f"\n{'=' * 70}")
        print(f"2.{idx} Processing: {image_name}")
        print(f"{'=' * 70}")


        print(f"\n2.{idx}.1 Replacing last 2 LSB with random values")
        manipulated_image = replace_2_lsb_random(original_image)


        print(f"\n2.{idx}.2 Visual quality check:")
        check_visual_quality(original_image, manipulated_image, image_name)


        print(f"\n2.{idx}.3 PDF comparison (plotting histograms)")
        compare_histograms(original_image, manipulated_image, image_name)

        print(f"\n2.{idx}.4 Statistical analysis:")
        detect_manipulation_analysis(original_image, manipulated_image, image_name)

        save_path = f'/Users/juliaszymanska/MachineLearning/{save_filename}'
        Image.fromarray(manipulated_image).save(save_path)
        print(f"\n✓ Manipulated image saved as '{save_filename}'")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)