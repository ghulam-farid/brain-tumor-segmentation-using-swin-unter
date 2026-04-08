import uuid
import cv2
from flask import Blueprint, request, jsonify, current_app
import torch
import numpy as np
import nibabel as nib
from werkzeug.utils import secure_filename
from utils.preprocessing import MRIPreprocessor
from utils.postprocessing import PostProcessor
from models.segmentation_model import BrainTumorSegmentationModel

upload_bp = Blueprint('upload', __name__, url_prefix='/api')

# Initialize model globally (cache it)
model = None


def get_model():
    global model
    if model is None:
        model = BrainTumorSegmentationModel(
            current_app.config['MODEL_PATH'],
            device=current_app.config['DEVICE']
        )
    return model


@upload_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
    }), 200


@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle MRI image upload and perform segmentation.

    Expected: Multipart form with 4 NIfTI files (T1, T1c, T2, FLAIR)
    """
    try:
        # Validate file presence
        required_files = ['t1', 't1c', 't2', 'flair']
        if not all(f in request.files for f in required_files):
            return jsonify({'error': 'Missing required MRI modalities'}), 400

        # Create unique session ID
        session_id = str(uuid.uuid4())
        session_upload_dir = current_app.config['UPLOAD_FOLDER'] / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save uploaded files
        print(f"[{session_id}] Processing upload...")
        modalities = {}

        for modality in required_files:
            file = request.files[modality]
            if file.filename == '':
                return jsonify({'error': f'No file for {modality}'}), 400

            filename = secure_filename(f"{modality}.nii.gz")
            filepath = session_upload_dir / filename
            file.save(filepath)
            modalities[modality] = filepath

        # 2. Preprocess images
        print(f"[{session_id}] Preprocessing images...")
        preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
        images = []
        original_shape = None

        for modality in required_files:
            img_array = preprocessor.load_nifti(str(modalities[modality]))

            # Save original shape for later
            if original_shape is None:
                original_shape = img_array.shape

            preprocessed = preprocessor.preprocess(img_array)
            images.append(preprocessed)
            print(f"[{session_id}] ✓ Preprocessed {modality}: shape {preprocessed.shape}")


        # Stack into 4-channel image: (H, W, D, C) -> (C, H, W, D)
        image_stack = np.stack(images, axis=0).astype(np.float32)
        print(f"[{session_id}] Image stack shape: {image_stack.shape}")

        # Add batch dimension: (1, 4, 128, 128, 128)
        image_tensor = torch.from_numpy(image_stack).unsqueeze(0)
        print(f"[{session_id}] Tensor shape: {image_tensor.shape}")

        # 3. Model inference
        print(f"[{session_id}] Running inference...")
        model = get_model()
        output = model.predict(image_tensor)
        print(f"[{session_id}] Model output shape: {output.shape}")

        segmentation = model.get_prediction_mask(output)
        print(f"[{session_id}] Segmentation shape: {segmentation.shape}")

        # 4. Post-processing
        print(f"[{session_id}] Post-processing...")
        post_processor = PostProcessor()
        segmentation = post_processor.post_process(segmentation)
        print(f"[{session_id}] Post-processed segmentation shape: {segmentation.shape}")

        # 5. Save results
        prediction_dir = current_app.config['PREDICTIONS_FOLDER'] / session_id
        prediction_dir.mkdir(parents=True, exist_ok=True)

        # Save segmentation mask
        seg_nii = nib.Nifti1Image(segmentation, np.eye(4))
        seg_path = prediction_dir / 'segmentation_mask.nii.gz'
        nib.save(seg_nii, str(seg_path))
        print(f"[{session_id}] ✓ Saved segmentation mask: {seg_path}")

        # Save visualization (axial slice at middle)
        slice_idx = segmentation.shape[-1] // 2  # Middle slice
        print(f"[{session_id}] Creating visualization at slice {slice_idx}...")

        # Get the T1 weighted image (first modality) for visualization
        mri_slice = image_stack[0, :, :, slice_idx]  # Shape: (128, 128)
        seg_slice = segmentation[:, :, slice_idx]  # Shape: (128, 128)

        print(f"[{session_id}] MRI slice shape: {mri_slice.shape}, Seg slice shape: {seg_slice.shape}")

        vis_image = visualize_segmentation(mri_slice, seg_slice)
        vis_path = prediction_dir / 'preview.png'
        cv2.imwrite(str(vis_path), vis_image)
        print(f"[{session_id}] ✓ Saved visualization: {vis_path}")

        # # Save visualization (first slice)
        # vis_image = visualize_segmentation(image_stack[0, :, :, 64], np.squeeze(segmentation)[:, :, 64])
        # vis_path = prediction_dir / 'preview.png'
        # cv2.imwrite(str(vis_path), vis_image)

        # 6. Calculate metrics
        # print(f"[{session_id}] Computing statistics...")
        # stats = compute_tumor_statistics(segmentation)
        # print(f"[{session_id}] Tumor statistics: {stats}")
        # 6. Calculate metrics
        print(f"[{session_id}] Computing statistics...")
        stats = post_processor.get_tumor_statistics(segmentation)
        print(f"[{session_id}] Tumor statistics: {stats}")

        print(f"[{session_id}] ✓ Complete!")

        return jsonify({
            'session_id': session_id,
            'status': 'success',
            'tumor_statistics': stats,
            'message': 'Segmentation complete'
        }), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        #print the error line number
        print(f"Error occurred at line {e.__traceback__.tb_lineno}")
        print(f"Error occurred in file {e.__traceback__.tb_frame.f_code.co_filename}")

        return jsonify({'error': str(e)}), 500


@upload_bp.route('/result/<session_id>', methods=['GET'])
def get_result(session_id):
    """Retrieve segmentation results."""
    try:
        prediction_dir = current_app.config['PREDICTIONS_FOLDER'] / session_id

        if not prediction_dir.exists():
            return jsonify({'error': 'Session not found'}), 404

        seg_path = prediction_dir / 'segmentation_mask.nii.gz'
        vis_path = prediction_dir / 'preview.png'

        if not seg_path.exists():
            return jsonify({'error': 'Results not ready'}), 404

        return jsonify({
            'session_id': session_id,
            'segmentation_url': f'/predictions/{session_id}/segmentation_mask.nii.gz',
            'preview_url': f'/predictions/{session_id}/preview.png'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def visualize_segmentation(mri_slice, seg_slice):
    """
    Create visualization of MRI with segmentation overlay.

    Args:
        mri_slice: (H, W) - 2D MRI slice
        seg_slice: (H, W) - 2D segmentation slice

    Returns:
        blended_image: (H, W, 3) - RGB visualization
    """
    try:
        print(f"[Visualization] MRI slice shape: {mri_slice.shape}, dtype: {mri_slice.dtype}")
        print(f"[Visualization] Seg slice shape: {seg_slice.shape}, dtype: {seg_slice.dtype}")

        # Ensure shapes are 2D
        mri_slice = np.squeeze(mri_slice)
        seg_slice = np.squeeze(seg_slice)

        if mri_slice.ndim != 2:
            raise ValueError(f"MRI slice must be 2D, got shape {mri_slice.shape}")
        if seg_slice.ndim != 2:
            raise ValueError(f"Seg slice must be 2D, got shape {seg_slice.shape}")

        # Normalize MRI for display
        mri_min = np.min(mri_slice)
        mri_max = np.max(mri_slice)
        mri_range = mri_max - mri_min

        if mri_range > 0:
            mri_normalized = (mri_slice - mri_min) / mri_range
        else:
            mri_normalized = np.zeros_like(mri_slice)

        mri_vis = (mri_normalized * 255).astype(np.uint8)
        print(f"[Visualization] MRI normalized: min={mri_vis.min()}, max={mri_vis.max()}")

        # Create BGR image (OpenCV format)
        rgb_image = cv2.cvtColor(mri_vis, cv2.COLOR_GRAY2BGR)
        overlay = rgb_image.copy()

        # Color mapping for segmentation (BGR format for OpenCV)
        colors = {
            1: (0, 0, 255),  # Red (NCR/NET)
            2: (0, 255, 0),  # Green (Edema)
            3: (255, 0, 0),  # Blue (ET)
        }

        # Apply colors to overlay
        for label_id, color in colors.items():
            mask = seg_slice == label_id

            if np.any(mask):
                print(f"[Visualization] Label {label_id}: {np.sum(mask)} voxels")
                overlay[mask] = color

        # Blend overlay with original
        blended_image = cv2.addWeighted(overlay, 0.6, rgb_image, 0.4, 0)

        print(f"[Visualization] Final image shape: {blended_image.shape}")
        return blended_image

    except Exception as e:
        print(f"[Visualization] Error: {e}")
        import traceback
        print(traceback.format_exc())
        raise


def compute_tumor_statistics(segmentation):
    """
    Compute tumor volume and statistics.

    Args:
        segmentation: (H, W, D) - 3D segmentation mask

    Returns:
        dict: Tumor statistics
    """
    voxel_volume = 1.0  # mm³ (for 1x1x1mm voxels)

    stats = {}
    labels = {
        1: 'Necrotic/Non-Enhancing Tumor (NCR/NET)',
        2: 'Peritumoral Edema (ED)',
        3: 'Gadolinium-Enhancing Tumor (ET)'
    }

    for label_id, label_name in labels.items():
        mask = segmentation == label_id
        voxel_count = np.sum(mask)
        volume = voxel_count * voxel_volume

        stats[label_name] = {
            'voxel_count': int(voxel_count),
            'volume_mm3': float(volume),
            'volume_cm3': float(volume / 1000)
        }

    return stats