import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2
import logging
import scipy.spatial.distance as sp_distance # For cosine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClothingFeatureExtractor:
    def __init__(self, resnet_weights=ResNet50_Weights.IMAGENET1K_V2):
        try:
            self.weights = resnet_weights
            self.model = resnet50(weights=self.weights)
            # Modify model to be a feature extractor (remove final FC layer)
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_extractor.eval()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor.to(self.device)

            self.transforms = self.weights.transforms()
            logging.info(f"ClothingFeatureExtractor: ResNet-50 feature extractor initialized on {self.device}.")
        except Exception as e:
            logging.error(f"Error initializing ResNet-50 for ClothingFeatureExtractor: {e}", exc_info=True)
            raise

    def _get_clothing_roi_on_crop(self, person_crop_shape, top_percent=0.20, bottom_percent=0.90):
        h, w = person_crop_shape[:2]
        roi_y1 = int(h * top_percent)
        roi_y2 = int(h * bottom_percent)
        roi_x1 = 0
        roi_x2 = w

        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            logging.warning(f"Calculated invalid clothing ROI ({roi_y1},{roi_y2},{roi_x1},{roi_x2}) for crop shape {person_crop_shape}. Using full crop.")
            return 0, h, 0, w
        return roi_y1, roi_y2, roi_x1, roi_x2

    def _extract_color_histogram(self, image_crop_bgr, bins_per_channel=16):
        if image_crop_bgr is None or image_crop_bgr.size == 0:
            logging.warning("_extract_color_histogram: Input image_crop_bgr is None or empty.")
            return None
        try:
            hist = cv2.calcHist([image_crop_bgr], [0, 1, 2], None,
                                [bins_per_channel, bins_per_channel, bins_per_channel],
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except Exception as e:
            logging.error(f"Error calculating color histogram: {e}", exc_info=True)
            return None

    def extract_clothing_features(self, person_crop_bgr):
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            logging.warning("extract_clothing_features: Input person_crop_bgr is None or empty.")
            return None

        roi_y1, roi_y2, roi_x1, roi_x2 = self._get_clothing_roi_on_crop(person_crop_bgr.shape)
        clothing_region_bgr = person_crop_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        if clothing_region_bgr.size == 0:
            logging.warning("Clothing ROI resulted in an empty image. Cannot extract features.")
            return None

        deep_features_vector = None
        try:
            clothing_region_rgb = cv2.cvtColor(clothing_region_bgr, cv2.COLOR_BGR2RGB)
            pil_image = T.ToPILImage()(clothing_region_rgb)
            img_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pooled_features = self.feature_extractor(img_tensor)
            deep_features_vector = torch.flatten(pooled_features, start_dim=1).cpu().numpy()
        except Exception as e:
            logging.error(f"Error extracting deep features for clothing: {e}", exc_info=True)

        color_hist_vector = self._extract_color_histogram(clothing_region_bgr)

        if deep_features_vector is None and color_hist_vector is None:
            logging.warning("Both deep feature and color histogram extraction failed for clothing.")
            return None

        return {'deep_features': deep_features_vector, 'color_histogram': color_hist_vector}

class ClothingComparator:
    @staticmethod
    def compare_clothing_features(features1, features2,
                                  deep_feature_weight=0.7, color_hist_weight=0.3,
                                  deep_feature_sim_threshold=0.6,
                                  hist_correlation_threshold=0.5):
        if not features1 or not features2:
            logging.warning("One or both feature sets are None. Cannot compare.")
            return 0.0, True

        df1 = features1.get('deep_features')
        ch1 = features1.get('color_histogram')
        df2 = features2.get('deep_features')
        ch2 = features2.get('color_histogram')

        deep_sim = 0.0
        hist_sim_for_decision = -1.0 # Use -1 as a "not similar" baseline for correlation

        df_available = False
        if df1 is not None and df2 is not None:
            try:
                df1_flat = df1.flatten()
                df2_flat = df2.flatten()
                if df1_flat.size == df2_flat.size and df1_flat.size > 0 and np.linalg.norm(df1_flat) > 0 and np.linalg.norm(df2_flat) > 0 :
                    cos_dist = sp_distance.cosine(df1_flat, df2_flat)
                    deep_sim = 1.0 - cos_dist
                    df_available = True
                    logging.debug(f"Deep feature cosine similarity: {deep_sim:.4f}")
                else:
                    logging.warning(f"Deep features have zero norm, mismatched shapes or empty. DF1 shape: {df1_flat.shape}, DF2 shape: {df2_flat.shape}. Cannot compute cosine similarity.")
                    deep_sim = 0.0
            except Exception as e:
                logging.error(f"Error calculating deep feature similarity: {e}", exc_info=True)
                deep_sim = 0.0
        else:
            logging.debug("One or both deep features are None.")

        ch_available = False
        if ch1 is not None and ch2 is not None:
            try:
                ch1_f32 = np.float32(ch1)
                ch2_f32 = np.float32(ch2)
                # Ensure histograms are not empty and have the same size
                if ch1_f32.size == ch2_f32.size and ch1_f32.size > 0:
                    hist_sim_for_decision = cv2.compareHist(ch1_f32, ch2_f32, cv2.HISTCMP_CORREL)
                    ch_available = True
                    logging.debug(f"Color histogram correlation: {hist_sim_for_decision:.4f}")
                else:
                    logging.warning(f"Color histograms are empty or mismatched. CH1 size: {ch1_f32.size}, CH2 size: {ch2_f32.size}. Cannot compare.")
                    hist_sim_for_decision = -1.0 # Worst correlation
            except Exception as e:
                logging.error(f"Error calculating color histogram similarity: {e}", exc_info=True)
                hist_sim_for_decision = -1.0
        else:
            logging.debug("One or both color histograms are None.")

        # Combined Similarity Score Calculation
        # Normalize hist_sim from [-1,1] to [0,1] for averaging: (val + 1) / 2
        hist_sim_for_avg = (hist_sim_for_decision + 1) / 2 if ch_available else 0.0

        current_deep_weight = deep_feature_weight if df_available else 0
        current_color_weight = color_hist_weight if ch_available else 0
        total_weight = current_deep_weight + current_color_weight

        if total_weight > 0:
            combined_similarity = (deep_sim * current_deep_weight + hist_sim_for_avg * current_color_weight) / total_weight
        else:
            combined_similarity = 0.0

        # Change Detection Logic
        change_detected = False
        # Prioritize deep features if available for change detection
        if df_available:
            if deep_sim < deep_feature_sim_threshold:
                logging.info(f"Clothing change DETECTED: Deep feature similarity {deep_sim:.4f} < threshold {deep_feature_sim_threshold:.4f}")
                change_detected = True
        # If deep features are not available or did not detect a change, check color histogram if available
        if not change_detected and ch_available:
            if hist_sim_for_decision < hist_correlation_threshold:
                logging.info(f"Clothing change DETECTED: Histogram correlation {hist_sim_for_decision:.4f} < threshold {hist_correlation_threshold:.4f}")
                change_detected = True

        # If neither feature type was available for comparison.
        if not df_available and not ch_available:
             logging.warning("No features (DF or CH) were available for comparison. Assuming change as a precaution.")
             change_detected = True # Default to change if no data

        return combined_similarity, change_detected

if __name__ == '__main__':
    logging.info("Starting ClothingFeatureExtractor example usage...")
    extractor = None
    try:
        extractor = ClothingFeatureExtractor()
    except Exception as e:
        logging.error(f"Failed to initialize ClothingFeatureExtractor: {e}. Example skipped.")

    if extractor:
        person_h, person_w = 256, 128
        dummy_person_crop_bgr = np.random.randint(0, 256, (person_h, person_w, 3), dtype=np.uint8)
        torso_y1, torso_y2 = int(person_h * 0.2), int(person_h * 0.7)
        torso_y1 = max(0, torso_y1)
        torso_y2 = min(person_h, torso_y2)
        if torso_y1 < torso_y2 :
            dummy_person_crop_bgr[torso_y1:torso_y2, :, :] = [255, 0, 0]

        logging.info(f"Extracting clothing features from dummy person crop (shape: {dummy_person_crop_bgr.shape})...")
        features = extractor.extract_clothing_features(dummy_person_crop_bgr)

        if features:
            if features['deep_features'] is not None:
                logging.info(f"  Deep features extracted, shape: {features['deep_features'].shape}")
                assert features['deep_features'].shape[1] == 2048
            else:
                logging.warning("  Deep features extraction returned None.")

            if features['color_histogram'] is not None:
                logging.info(f"  Color histogram extracted, shape: {features['color_histogram'].shape}")
                assert features['color_histogram'].shape[0] == (16**3)
            else:
                logging.warning("  Color histogram extraction returned None.")
        else:
            logging.error("Clothing feature extraction failed for the dummy crop.")

        logging.info("Testing with None input...")
        assert extractor.extract_clothing_features(None) is None
        logging.info("Testing with empty input...")
        empty_person_crop = np.array([], dtype=np.uint8).reshape(0,0,3)
        assert extractor.extract_clothing_features(empty_person_crop) is None

        logging.info("Testing with very small crop (10x10)...")
        small_crop = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        small_features = extractor.extract_clothing_features(small_crop)

        if small_features:
            logging.info(f"  Small crop features extracted. DF: {small_features['deep_features'] is not None}, CH: {small_features['color_histogram'] is not None}")
            if small_features['deep_features'] is not None:
                 assert small_features['deep_features'].shape[1] == 2048
            if small_features['color_histogram'] is not None:
                 assert small_features['color_histogram'].shape[0] == (16**3)
        else:
            logging.info("  Small crop feature extraction returned None.")

        logging.info("\n--- Testing ClothingComparator ---")
        dummy_features1 = extractor.extract_clothing_features(dummy_person_crop_bgr)

        dummy_person_crop_bgr2 = dummy_person_crop_bgr.copy()
        if torso_y1 < torso_y2 : # Ensure torso region is valid before assignment
            dummy_person_crop_bgr2[torso_y1:torso_y2, :, :] = [0, 0, 255] # Red torso
        dummy_features2_changed = extractor.extract_clothing_features(dummy_person_crop_bgr2)

        if dummy_features1: # Ensure features1 is not None
            sim_same, change_same = ClothingComparator.compare_clothing_features(dummy_features1, dummy_features1)
            logging.info(f"Comparison (same features): Similarity={sim_same:.4f}, Change Detected={change_same}")
            assert not change_same, "Comparing same features should not detect a change."
            if sim_same <= 0.9 and (dummy_features1['deep_features'] is not None or dummy_features1['color_histogram'] is not None) : # only assert if features were actually comparable
                 logging.warning(f"Similarity for same features was {sim_same:.4f}, expected > 0.9. Check feature stability.")


        if dummy_features1 and dummy_features2_changed: # Ensure both are not None
            sim_diff, change_diff = ClothingComparator.compare_clothing_features(dummy_features1, dummy_features2_changed)
            logging.info(f"Comparison (different features): Similarity={sim_diff:.4f}, Change Detected={change_diff}")
            assert change_diff, "Comparing different features should detect a change."

        # Test 3: Features with Nones (more granular)
        logging.info("--- Testing Comparator with Partial/None Features ---")
        valid_df1 = dummy_features1['deep_features'] if dummy_features1 and dummy_features1['deep_features'] is not None else np.random.rand(1,2048).astype(np.float32)
        valid_ch1 = dummy_features1['color_histogram'] if dummy_features1 and dummy_features1['color_histogram'] is not None else np.random.rand(16**3).astype(np.float32)

        features_df_only = {'deep_features': valid_df1, 'color_histogram': None}
        features_ch_only = {'deep_features': None, 'color_histogram': valid_ch1}
        features_both_none = {'deep_features': None, 'color_histogram': None}

        # DF vs CH only (should rely on what's available, or lack thereof)
        sim_df_vs_ch, change_df_vs_ch = ClothingComparator.compare_clothing_features(features_df_only, features_ch_only)
        logging.info(f"Comparison (DF only vs CH only): Similarity={sim_df_vs_ch:.4f}, Change Detected={change_df_vs_ch}")
        assert change_df_vs_ch, "Comparing DF-only to CH-only should detect change (no common features)."

        # Both features None vs valid features (should detect change)
        sim_none_vs_valid, change_none_vs_valid = ClothingComparator.compare_clothing_features(features_both_none, dummy_features1 if dummy_features1 else features_df_only)
        logging.info(f"Comparison (Both None vs Valid): Similarity={sim_none_vs_valid:.4f}, Change Detected={change_none_vs_valid}")
        assert change_none_vs_valid, "Comparing None features to valid features should detect change."

        # One feature set is entirely None
        sim_one_none, change_one_none = ClothingComparator.compare_clothing_features(dummy_features1, None)
        logging.info(f"Comparison (Valid vs None Set): Similarity={sim_one_none:.4f}, Change Detected={change_one_none}")
        assert change_one_none, "Comparing valid features to a None set should detect change."

    logging.info("ClothingFeatureExtractor & Comparator example usage finished.")
