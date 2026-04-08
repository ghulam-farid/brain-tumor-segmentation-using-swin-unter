import torch
from monai.networks.nets import SwinUNETR
from pathlib import Path
import numpy as np

class BrainTumorSegmentationModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = Path(model_path)
        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=3,  # MUST match training
            feature_size=24,  # MUST match training
            use_checkpoint=False  # inference mode
        )
        return model.to(self.device)

    def _load_weights(self):
        """Load pre-trained weights."""
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded from {self.model_path}")
        else:
            print(f"⚠ Model not found at {self.model_path}")

    def predict(self, image_tensor):
        """
        Perform inference.

        Args:
            image_tensor: (1, 4, 128, 128, 128) - batch of preprocessed images

        Returns:
            segmentation: (1, 4, 128, 128, 128) - model output
        """
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)

        return output.cpu()

    # def get_prediction_mask(self, output):
    #     """Convert model output to segmentation mask."""
    #     # Take argmax across channel dimension
    #     segmentation = torch.argmax(output, dim=1).squeeze(0).numpy()
    #     return segmentation.astype(np.uint8)
    def get_prediction_mask(self, output):
        """
        Convert model output to segmentation mask.

        Args:
            output: (1, 4, H, W, D) - Model output

        Returns:
            segmentation: (H, W, D) - Single-channel segmentation mask
        """
        # Take argmax across channel dimension
        # output shape: (1, 4, 128, 128, 128)
        # segmentation shape: (1, 128, 128, 128)
        segmentation = torch.argmax(output, dim=1).squeeze(0).numpy()

        # Now shape is: (128, 128, 128)
        return segmentation.astype(np.uint8)