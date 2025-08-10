import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path):
    """
    Extracts color (HSV histogram) and texture (GLCM) features from a single image.
    """
    try:
        # Read and resize image to a standard size
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None
        
        img_resized = cv2.resize(img, (200, 200))

        # 1. Color Features (HSV Color Histogram)
        hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
        cv2.normalize(h_hist, h_hist)
        cv2.normalize(s_hist, s_hist)
        cv2.normalize(v_hist, v_hist)
        color_features = np.concatenate((h_hist.flatten(), s_hist.flatten(), v_hist.flatten()))

        # 2. Texture Features (GLCM)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])

        # Combine all features into a single vector
        combined_features = np.concatenate((color_features, texture_features))
        
        return combined_features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None