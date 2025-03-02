import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
import os

def count_coins(image_path, output_folder):
    # Load the image
    image_original = cv2.imread(image_path)
    # Create a copy for contour drawing
    image_contours = image_original.copy()
    # Convert images from BGR to RGB (for visualization)
    image_contours = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)
    image_display = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    image_blurred = cv2.medianBlur(image_gray, 7)
    # Apply thresholding for binary conversion
    X, image_binary = cv2.threshold(image_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological operation to clean up noise
    morph_kernel = np.ones((3, 3), np.uint8)
    image_cleaned = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    # Labeling connected components for segmentation
    labeled_image = measure.label(image_cleaned, connectivity=2)
    image_segmented = color.label2rgb(labeled_image, bg_label=0)

    # Find contours
    contours, _ = cv2.findContours(image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw detected contours on the image
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save detected images
    detected_path = os.path.join(output_folder, "detected_coins.jpg")
    segmented_path = os.path.join(output_folder, "segmented_coins.jpg")

    cv2.imwrite(detected_path, cv2.cvtColor(image_contours, cv2.COLOR_RGB2BGR))
    cv2.imwrite(segmented_path, cv2.cvtColor((image_segmented * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(18, 6))  # Wider figure for better visibility

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_display)
    plt.axis("off")  # Hide axes for better visualization

    plt.subplot(1, 3, 2)
    plt.title("Detected Coins")
    plt.imshow(image_contours)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Segmented Image")
    plt.imshow(image_segmented)
    plt.axis("off")

    plt.show()


    return len(contours)

# Define input and output paths
input_image = "../input_IMAGES/coins.jpg"
output_directory = "../output_IMAGES/"

# Run detection
coins_detected = count_coins(input_image, output_directory)
print(f"{coins_detected} coins Detected ")

