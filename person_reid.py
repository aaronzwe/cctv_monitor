import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2 # For image operations if needed, e.g. color conversion
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersonReID:
    def __init__(self):
        '''
        Initializes the PersonReID class with a pre-trained ResNet-50 model.
        '''
        try:
            # Load pre-trained ResNet-50 model with the latest recommended weights
            self.weights = ResNet50_Weights.IMAGENET1K_V2 # Using V2 for potentially better features
            self.model = resnet50(weights=self.weights)
            self.model.eval()  # Set the model to evaluation mode

            # Check for GPU availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logging.info(f"ResNet-50 model loaded successfully on {self.device}.")

            # Define the image transformations based on the weights used
            self.transforms = self.weights.transforms()
            # The transforms() method from IMAGENET1K_V2 typically includes:
            # Resize(256, interpolation=InterpolationMode.BILINEAR)
            # CenterCrop(224)
            # ToTensor()
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            logging.info(f"Image transformations for ResNet-50 initialized: {self.transforms}")

        except Exception as e:
            logging.error(f"Error loading ResNet-50 model: {e}", exc_info=True)
            raise

    def extract_features(self, person_image_np):
        '''
        Extracts deep features from a given person image.
        :param person_image_np: The input image of a person (NumPy array, BGR format from OpenCV).
        :return: A NumPy array representing the feature vector, or None if extraction fails.
        '''
        if person_image_np is None or person_image_np.size == 0:
            logging.warning("Input person_image is None or empty.")
            return None

        try:
            # Convert BGR (OpenCV default) to RGB
            person_image_rgb = cv2.cvtColor(person_image_np, cv2.COLOR_BGR2RGB)

            # Convert NumPy array to PIL Image, as torchvision transforms often expect PIL Image
            # Or, if transforms handle tensors, convert to tensor first then apply some transforms
            # The standard `weights.transforms()` expects a PIL image or a tensor.
            # Let's convert to PIL Image first.
            pil_image = T.ToPILImage()(person_image_rgb)

            # Apply transformations
            img_tensor = self.transforms(pil_image)

            # Add batch dimension and send to device
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad(): # Ensure no gradients are computed
                features = self.model(img_tensor)

            # The output of ResNet-50 before the final FC layer is typically used for features.
            # However, the default `self.model(img_tensor)` gives classification scores.
            # To get features, we need to hook into an earlier layer or modify the model.
            # A common approach is to remove the final fully connected layer (classifier).

            # Let's modify the model to be a feature extractor by removing the fc layer
            # We can do this once in __init__ or use a hook.
            # For simplicity, if we want the output of the avgpool layer:
            feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1]) # Remove the last layer (fc)
            feature_extractor.to(self.device) # Ensure it's on the correct device
            feature_extractor.eval() # Set to eval mode

            with torch.no_grad():
                pooled_features = feature_extractor(img_tensor) # This will be (batch_size, feature_dim, 1, 1)

            # Flatten the features
            feature_vector = torch.flatten(pooled_features, start_dim=1) # Flatten all dims except batch

            return feature_vector.cpu().numpy() # Convert to NumPy array on CPU

        except Exception as e:
            logging.error(f"Error during feature extraction: {e}", exc_info=True)
            return None

if __name__ == '__main__':
    logging.info("Starting PersonReID example usage...")

    reid_model = None
    try:
        reid_model = PersonReID()
        logging.info("PersonReID model initialized successfully.")
    except Exception as e:
        logging.error(f"Could not initialize PersonReID: {e}. Skipping example.")
        # exit() # Avoid exit in automated test; allow to proceed if possible or fail naturally

    if reid_model:
        # Create a dummy person image (e.g., a small colored patch)
        dummy_person_height, dummy_person_width = 128, 64 # Typical aspect ratio for person crops
        # Create a BGR image as if it came from OpenCV
        dummy_person_bgr = np.random.randint(0, 256, (dummy_person_height, dummy_person_width, 3), dtype=np.uint8)
        cv2.putText(dummy_person_bgr, "Person", (5, dummy_person_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        logging.info(f"Extracting features from a dummy person image of shape: {dummy_person_bgr.shape}")
        features = reid_model.extract_features(dummy_person_bgr)

        if features is not None:
            logging.info(f"Successfully extracted features. Feature vector shape: {features.shape}")
            # features will be a 2D array (1, num_features), e.g. (1, 2048) for ResNet-50 avgpool output
            assert features.shape[0] == 1 and features.shape[1] == 2048, \
                f"Expected feature shape (1, 2048), got {features.shape}"
            logging.info(f"Feature extraction example successful. First 5 feature values: {features[0, :5]}")
        else:
            logging.error("Feature extraction failed for the dummy image.")

        # Test with an empty or None image
        logging.info("Testing feature extraction with None input...")
        none_features = reid_model.extract_features(None)
        assert none_features is None, "Feature extraction with None input should return None."
        logging.info("Test with None input successful.")

        logging.info("Testing feature extraction with empty NumPy array...")
        empty_features = reid_model.extract_features(np.array([]))
        assert empty_features is None, "Feature extraction with empty input should return None."
        logging.info("Test with empty input successful.")


    logging.info("PersonReID example usage finished.")
