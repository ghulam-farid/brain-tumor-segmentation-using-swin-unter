import numpy as np
import nibabel as nib
from scipy import ndimage
import cv2
from pathlib import Path


class MRIPreprocessor:
    def __init__(self, target_shape=(128, 128, 128)):
        self.target_shape = target_shape

    def load_nifti(self, filepath):
        """Load NIfTI file."""
        img = nib.load(filepath)
        return np.array(img.dataobj)

    def z_score_normalization(self, image, mask=None):
        """Apply z-score normalization."""
        if mask is not None:
            mean = np.mean(image[mask > 0])
            std = np.std(image[mask > 0])
        else:
            mean = np.mean(image)
            std = np.std(image)

        if std == 0:
            return image
        return (image - mean) / std

    def resize_3d(self, image, target_shape):
        """Resize 3D image to target shape."""
        current_shape = image.shape
        factors = np.array(target_shape) / np.array(current_shape)

        # Use scipy zoom for 3D
        return ndimage.zoom(image, factors, order=1)

    def preprocess(self, image_array):
        """Complete preprocessing pipeline."""
        # 1. Normalize
        normalized = self.z_score_normalization(image_array)

        # 2. Resize to target shape
        resized = self.resize_3d(normalized, self.target_shape)

        # 3. Clip extreme values
        resized = np.clip(resized, -3, 3)

        # 4. Convert to float32
        resized = resized.astype(np.float32)

        return resized

    def get_brain_mask(self, image):
        """Create brain mask using Otsu's thresholding."""
        # Simple threshold-based mask
        threshold = np.percentile(image, 5)
        mask = image > threshold
        return mask


# class PostProcessor:
#     def __init__(self):
#         self.labels = {
#             0: 'Background',
#             1: 'Necrotic/Non-Enhancing Tumor (NCR/NET)',
#             2: 'Edema',
#             3: 'Enhancing Tumor (ET)'
#         }
#
#     def remove_small_components(self, segmentation, min_size=100):
#         """Remove small connected components (noise)."""
#         for label in np.unique(segmentation)[1:]:
#             mask = segmentation == label
#             labeled, num_features = ndimage.label(mask)
#             component_sizes = np.bincount(labeled.ravel())
#
#             for component_id in range(1, num_features + 1):
#                 if component_sizes[component_id] < min_size:
#                     segmentation[labeled == component_id] = 0
#
#         return segmentation
#
#     def morphological_closing(self, segmentation):
#         """Apply morphological closing to smooth segmentation."""
#         from scipy.ndimage import binary_closing
#         processed = segmentation.copy()
#
#         for label in np.unique(segmentation)[1:]:
#             mask = segmentation == label
#             closed = binary_closing(mask, structure=np.ones((3, 3, 3)))
#             processed[closed] = label
#
#         return processed
#
#     def post_process(self, segmentation):
#         """Apply post-processing steps."""
#         # 1. Remove small components
#         segmentation = self.remove_small_components(segmentation, min_size=50)
#
#         # 2. Morphological closing
#         segmentation = self.morphological_closing(segmentation)
#
#         return segmentation