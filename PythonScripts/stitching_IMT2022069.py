import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error in loading image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb, image_gray

def detect_features(image_rgb, image_gray, output_folder, image_name):
    """Detects keypoints and computes descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    keypoint_image = cv2.drawKeypoints(image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoint_path = os.path.join(output_folder, f"keypoints_{image_name}.jpg")
    cv2.imwrite(keypoint_path, cv2.cvtColor(keypoint_image, cv2.COLOR_RGB2BGR))
    return keypoints, descriptors, keypoint_path

def match_features(desc1, desc2):
    """Matches features using BFMatcher with cross-check."""
    M = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    match = M.match(desc1, desc2)
    return sorted(match, key=lambda x: x.distance)

def visualize_matches(image1, kp1, image2, kp2, matches, output_folder):
    """Draws and saves the best matches."""
    image_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    path = os.path.join(output_folder, "image_matches.jpg")
    cv2.imwrite(path, cv2.cvtColor(image_matches, cv2.COLOR_RGB2BGR))
    return path

def compute_homography(kp1, kp2, matches, max_matches=50):
    """Computes the homography matrix using the best matches."""
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography!")
    best_match = matches[:max_matches]
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_match]).reshape(-1, 1, 2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_match]).reshape(-1, 1, 2)
    Homo, K = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return Homo

def stitch_images(image1, image2, H):
    """Warps image1 to align with image2 and blends them."""
    h2, w2, k2 = image2.shape
    W = cv2.warpPerspective(image1, H, (w2 * 2, h2))
    W[0:h2, 0:w2] = image2  # Overlay second image
    return W

def crop_black_regions(image):
    """Crops out black regions from the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    T, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# Define input and output directories
input_folder = "../input_IMAGES/"
output_folder = "../output_IMAGES/"
os.makedirs(output_folder, exist_ok=True)

# Load images
image1, gray1 = load_and_preprocess(os.path.join(input_folder, "Right_Side.jpg"))
image2, gray2 = load_and_preprocess(os.path.join(input_folder, "Left_Side.jpg"))

# Detect features and save keypoints
kp1, desc1, keypoint_img1 = detect_features(image1, gray1, output_folder, "image1")
kp2, desc2, keypoint_img2 = detect_features(image2, gray2, output_folder, "image2")

# Match features
feature_matches = match_features(desc1, desc2)

# Visualize and save matches
matched_image_path = visualize_matches(image1, kp1, image2, kp2, feature_matches, output_folder)

# Compute homography
homography_matrix = compute_homography(kp1, kp2, feature_matches)

# Stitch images
stitched_result = stitch_images(image1, image2, homography_matrix)

# Crop black regions
final_stitched_output = crop_black_regions(stitched_result)

# Save final stitched image
stitched_file_path = os.path.join(output_folder, "stitched_panorama.jpg")
cv2.imwrite(stitched_file_path, cv2.cvtColor(final_stitched_output, cv2.COLOR_RGB2BGR))

# Display results
plt.figure(figsize=(18, 10))  # Adjusted plot size

plt.subplot(2, 2, 1)
plt.imshow(cv2.imread(keypoint_img1))
plt.title("Keypoints in First Image")
plt.axis("off")  # Hide axes for better visualization

plt.subplot(2, 2, 2)
plt.imshow(cv2.imread(keypoint_img2))
plt.title("Keypoints in Second Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.imread(matched_image_path))
plt.title("Feature Matches")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(final_stitched_output)
plt.title("Final Stitched Panorama")
plt.axis("off")

plt.show()
