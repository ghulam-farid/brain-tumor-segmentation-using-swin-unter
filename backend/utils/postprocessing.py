# backend/utils/postprocessing.py
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, binary_dilation, binary_erosion
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class PostProcessor:
    """
    Post-processing module for brain tumor segmentation masks.
    Includes noise removal, morphological operations, and refinement.
    """

    def __init__(self):
        self.labels = {
            0: 'Background',
            1: 'Necrotic/Non-Enhancing Tumor (NCR/NET)',
            2: 'Peritumoral Edema (ED)',
            3: 'Gadolinium-Enhancing Tumor (ET)'
        }

        self.colors = {
            1: (255, 0, 0),  # NCR/NET - Red
            2: (0, 255, 0),  # Edema - Green
            3: (0, 0, 255),  # ET - Blue
        }

    # ==================== Noise Removal ====================

    def remove_small_components(self, segmentation, min_size=100):
        """
        Remove small connected components (noise) from segmentation.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            min_size: Minimum voxel count to keep a component

        Returns:
            segmentation: Cleaned 3D segmentation mask
        """
        processed = segmentation.copy().astype(np.int32)

        # Process each label separately
        for label_id in np.unique(segmentation)[1:]:  # Skip background (0)
            mask = segmentation == label_id
            labeled_array, num_features = ndimage.label(mask)

            if num_features == 0:
                continue

            component_sizes = np.bincount(labeled_array.ravel())

            # Remove small components
            for component_id in range(1, num_features + 1):
                if component_id < len(component_sizes) and component_sizes[component_id] < min_size:
                    processed[labeled_array == component_id] = 0

        return processed.astype(np.uint8)

    def remove_isolated_voxels(self, segmentation, connectivity=26):
        """
        Remove isolated voxels using connectivity check.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            connectivity: 6 (face), 18 (edge), or 26 (corner)

        Returns:
            segmentation: Cleaned mask with isolated voxels removed
        """
        processed = segmentation.copy()

        # Define connectivity structure
        if connectivity == 6:
            struct = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            struct = ndimage.generate_binary_structure(3, 2)
        else:  # 26
            struct = ndimage.generate_binary_structure(3, 3)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            # Count neighbors for each voxel
            try:
                neighbor_count = ndimage.convolve(
                    mask.astype(np.float32),
                    struct.astype(np.float32)
                )

                # Isolated voxels have only 1 neighbor (themselves)
                isolated = (neighbor_count <= 1) & mask
                processed[isolated] = 0
            except Exception as e:
                print(f"Warning in remove_isolated_voxels: {e}")
                continue

        return processed

    # ==================== Morphological Operations ====================

    def morphological_closing(self, segmentation, radius=2):
        """
        Apply morphological closing to fill small holes.
        Closing = Dilation followed by Erosion

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            radius: Radius of the structuring element

        Returns:
            segmentation: Closed mask
        """
        processed = segmentation.copy()
        struct = ndimage.generate_binary_structure(3, 2)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            try:
                # Apply closing
                closed = binary_closing(mask, structure=struct, iterations=radius)
                processed[closed] = label_id
                processed[~closed & mask] = 0
            except Exception as e:
                print(f"Warning in morphological_closing for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    def morphological_opening(self, segmentation, radius=1):
        """
        Apply morphological opening to remove small objects.
        Opening = Erosion followed by Dilation

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            radius: Radius of the structuring element

        Returns:
            segmentation: Opened mask
        """
        processed = segmentation.copy()
        struct = ndimage.generate_binary_structure(3, 2)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            try:
                # Apply opening
                opened = binary_opening(mask, structure=struct, iterations=radius)
                processed[opened] = label_id
                processed[~opened & mask] = 0
            except Exception as e:
                print(f"Warning in morphological_opening for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    def morphological_dilation(self, segmentation, iterations=1):
        """
        Apply morphological dilation to expand tumor regions.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            iterations: Number of dilation iterations

        Returns:
            segmentation: Dilated mask
        """
        processed = segmentation.copy()
        struct = ndimage.generate_binary_structure(3, 2)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            try:
                dilated = binary_dilation(mask, structure=struct, iterations=iterations)
                processed[dilated] = label_id
            except Exception as e:
                print(f"Warning in morphological_dilation for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    def morphological_erosion(self, segmentation, iterations=1):
        """
        Apply morphological erosion to shrink tumor regions.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            iterations: Number of erosion iterations

        Returns:
            segmentation: Eroded mask
        """
        processed = segmentation.copy()
        struct = ndimage.generate_binary_structure(3, 2)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            try:
                eroded = binary_erosion(mask, structure=struct, iterations=iterations)
                processed[eroded] = label_id
            except Exception as e:
                print(f"Warning in morphological_erosion for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    # ==================== Refinement Operations ====================

    def fill_holes(self, segmentation):
        """
        Fill holes (internal cavities) in each tumor region.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask

        Returns:
            segmentation: Mask with filled holes
        """
        processed = segmentation.copy()

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id

            if not np.any(mask):
                continue

            try:
                # Fill holes using binary_fill_holes
                filled = ndimage.binary_fill_holes(mask)
                processed[filled] = label_id
            except Exception as e:
                print(f"Warning in fill_holes for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    def keep_largest_component(self, segmentation):
        """
        Keep only the largest connected component for each label.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask

        Returns:
            segmentation: Mask with only largest components
        """
        processed = np.zeros_like(segmentation)

        for label_id in np.unique(segmentation)[1:]:
            mask = segmentation == label_id
            labeled_array, num_features = ndimage.label(mask)

            if num_features == 0:
                continue

            try:
                # Find largest component
                component_sizes = np.bincount(labeled_array.ravel())
                largest_component_id = np.argmax(component_sizes[1:]) + 1

                # Keep only largest component
                processed[labeled_array == largest_component_id] = label_id
            except Exception as e:
                print(f"Warning in keep_largest_component for label {label_id}: {e}")
                continue

        return processed.astype(np.uint8)

    def smooth_boundaries(self, segmentation, iterations=2):
        """
        Smooth boundaries using closing followed by opening.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            iterations: Number of iterations

        Returns:
            segmentation: Smoothed mask
        """
        processed = segmentation.copy()

        try:
            for _ in range(iterations):
                processed = self.morphological_closing(processed, radius=1)
                processed = self.morphological_opening(processed, radius=1)
        except Exception as e:
            print(f"Warning in smooth_boundaries: {e}")

        return processed

    # ==================== Label-Specific Refinement ====================

    def refine_edema_region(self, segmentation):
        """
        Refine edema (label 2) region by removing isolated components.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask

        Returns:
            segmentation: Refined mask
        """
        processed = segmentation.copy()

        try:
            # Keep only edema label
            edema_mask = segmentation == 2

            if np.sum(edema_mask) > 0:
                # Keep largest component of edema
                labeled_array, num_features = ndimage.label(edema_mask)
                if num_features > 1:
                    component_sizes = np.bincount(labeled_array.ravel())
                    if len(component_sizes) > 1:
                        largest_id = np.argmax(component_sizes[1:]) + 1
                        processed[edema_mask] = 0
                        processed[labeled_array == largest_id] = 2
        except Exception as e:
            print(f"Warning in refine_edema_region: {e}")

        return processed.astype(np.uint8)

    def ensure_tumor_hierarchy(self, segmentation):
        """
        Ensure proper tumor hierarchy:
        - ET (3) should be within TC (tumor core = 1+3)
        - TC should be within WT (whole tumor = 1+2+3)

        Args:
            segmentation: (H, W, D) - 3D segmentation mask

        Returns:
            segmentation: Corrected mask with proper hierarchy
        """
        processed = segmentation.copy()

        try:
            # Get masks for each region
            et_mask = segmentation == 3  # Enhancing Tumor
            net_mask = segmentation == 1  # NCR/NET
            edema_mask = segmentation == 2  # Edema

            # TC = ET + NCR/NET
            tc_mask = et_mask | net_mask

            # WT = TC + Edema
            wt_mask = tc_mask | edema_mask

            # Ensure ET is within TC
            processed[et_mask & ~tc_mask] = 0

            # Ensure TC is within WT
            processed[tc_mask & ~wt_mask] = 0
        except Exception as e:
            print(f"Warning in ensure_tumor_hierarchy: {e}")

        return processed.astype(np.uint8)

    # ==================== Main Post-Processing Pipeline ====================

    def post_process(self, segmentation,
                     remove_small_components=True,
                     min_component_size=100,
                     remove_isolated=True,
                     morphological_smoothing=True,
                     fill_holes_flag=True,
                     keep_largest=False,
                     enforce_hierarchy=True):
        """
        Complete post-processing pipeline for segmentation masks.

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            remove_small_components: Remove small connected components
            min_component_size: Minimum voxels for a component
            remove_isolated: Remove isolated voxels
            morphological_smoothing: Apply morphological smoothing
            fill_holes_flag: Fill holes in tumor regions
            keep_largest: Keep only largest component per label
            enforce_hierarchy: Ensure tumor hierarchy

        Returns:
            segmentation: Fully post-processed mask
        """
        try:
            processed = segmentation.copy().astype(np.uint8)

            print("[PostProcessor] Starting post-processing pipeline...")

            # Step 1: Remove small components
            if remove_small_components:
                print(f"[PostProcessor] Removing components < {min_component_size} voxels...")
                processed = self.remove_small_components(
                    processed,
                    min_size=min_component_size
                )

            # Step 2: Remove isolated voxels
            if remove_isolated:
                print("[PostProcessor] Removing isolated voxels...")
                processed = self.remove_isolated_voxels(processed)

            # Step 3: Fill holes
            if fill_holes_flag:
                print("[PostProcessor] Filling holes...")
                processed = self.fill_holes(processed)

            # Step 4: Morphological smoothing
            if morphological_smoothing:
                print("[PostProcessor] Smoothing boundaries...")
                processed = self.smooth_boundaries(processed, iterations=2)

            # Step 5: Keep only largest component (optional)
            if keep_largest:
                print("[PostProcessor] Keeping largest component...")
                processed = self.keep_largest_component(processed)

            # Step 6: Refine edema region
            print("[PostProcessor] Refining edema region...")
            processed = self.refine_edema_region(processed)

            # Step 7: Enforce tumor hierarchy
            if enforce_hierarchy:
                print("[PostProcessor] Enforcing tumor hierarchy...")
                processed = self.ensure_tumor_hierarchy(processed)

            print("[PostProcessor] Post-processing complete!")
            return processed.astype(np.uint8)

        except Exception as e:
            print(f"[PostProcessor] Error during post-processing: {e}")
            return segmentation.astype(np.uint8)

    # ==================== Utility Functions ====================

    def get_tumor_statistics(self, segmentation, voxel_spacing=(1, 1, 1)):
        """
        Calculate tumor statistics (volume, surface area, etc.).

        Args:
            segmentation: (H, W, D) - 3D segmentation mask
            voxel_spacing: (dx, dy, dz) - Physical spacing of voxels in mm

        Returns:
            dict: Tumor statistics for each label
        """
        stats = {}
        voxel_volume = np.prod(voxel_spacing)

        try:
            for label_id in np.unique(segmentation)[1:]:
                mask = segmentation == label_id
                voxel_count = np.sum(mask)
                volume_mm3 = voxel_count * voxel_volume
                volume_cm3 = volume_mm3 / 1000

                # Calculate surface area (approximation)
                if voxel_count > 0:
                    try:
                        grad_x = np.abs(np.gradient(mask.astype(float), axis=0))
                        grad_y = np.abs(np.gradient(mask.astype(float), axis=1))
                        grad_z = np.abs(np.gradient(mask.astype(float), axis=2))
                        surface_voxels = np.sum((grad_x + grad_y + grad_z) > 0)
                        surface_area = surface_voxels * (voxel_spacing[0] * voxel_spacing[1])
                    except:
                        surface_area = 0
                else:
                    surface_area = 0

                label_name = self.labels.get(label_id, f'Label {label_id}')

                stats[label_name] = {
                    'voxel_count': int(voxel_count),
                    'volume_mm3': float(volume_mm3),
                    'volume_cm3': float(volume_cm3),
                    'surface_area_mm2': float(surface_area),
                    'label_id': int(label_id)
                }
        except Exception as e:
            print(f"[PostProcessor] Error calculating statistics: {e}")

        return stats

    def get_label_info(self, label_id):
        """
        Get information about a specific tumor label.

        Args:
            label_id: Label ID (0-3)

        Returns:
            dict: Label name and color
        """
        return {
            'name': self.labels.get(label_id, 'Unknown'),
            'color': self.colors.get(label_id, (0, 0, 0)),
            'id': label_id
        }

    def get_all_labels_info(self):
        """
        Get information about all tumor labels.

        Returns:
            dict: All labels with their names and colors
        """
        return {
            'labels': self.labels,
            'colors': self.colors,
            'descriptions': {
                0: 'Background - Non-tumor tissue',
                1: 'Necrotic/Non-Enhancing Tumor Core - Dead tumor cells',
                2: 'Peritumoral Edema - Brain swelling around tumor',
                3: 'Gadolinium-Enhancing Tumor - Active tumor region'
            }
        }