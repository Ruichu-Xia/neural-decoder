from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvmodels
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import open_clip as clip
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from scipy.stats import binom
import scipy.spatial.distance
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance as FID 

from configs.config import config


class ImageEvaluationDataset(Dataset):
    """Dataset for loading images for evaluation."""
    
    def __init__(self, data_path: Path, 
                 prefix: str = '', 
                 model_name: str = 'clip', 
                 num_images: int = 200,
                 image_size: int = 224,
                 paired_data_path: Path | None = None):
        self.data_path = Path(data_path)
        self.prefix = prefix
        self.model_name = model_name
        self.num_images = num_images
        self.image_size = image_size
        self.paired_data_path = paired_data_path
        
        if model_name == 'clip':
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        else:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
    
    def __len__(self) -> int:
        return self.num_images
    
    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        img_path = self.data_path / f"{self.prefix}{idx}.png"
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img).float()
        img = F.resize(img, [self.image_size, self.image_size])
        img = self.normalize(img)
        
        if self.paired_data_path is not None:
            paired_img_path = self.paired_data_path / f"{self.prefix}{idx}.png"
            paired_img = Image.open(paired_img_path).convert('RGB')
            paired_img = F.to_tensor(paired_img).float()
            paired_img = F.resize(paired_img, [self.image_size, self.image_size])
            paired_img = self.normalize(paired_img)
            return img, paired_img
            
        return img


def prep_image_for_display(batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    """
    Denormalizes a batch of images.
    Args:
        batch (torch.Tensor): A batch of normalized images (B, C, H, W).
        mean (torch.Tensor): The mean used for normalization (C,).
        std (torch.Tensor): The standard deviation used for normalization (C,).
    Returns:
        torch.Tensor: Denormalized batch of images, clamped to [0, 1] (B, C, H, W).
    """
    denormalized_batch = batch * std + mean
    denormalized_batch = torch.clamp(denormalized_batch, 0, 1)
    displayable_batch_np = (denormalized_batch * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return displayable_batch_np


def display_image_batch(batch_np: np.ndarray, title: str = "Image Batch", max_images: int = 16):
    """
    Displays a batch of images using matplotlib.
    Assumes input batch is a NumPy array (B, H, W, C) in [0, 255] range and uint8 type.
    Args:
        batch_np (np.ndarray): Denormalized batch of images as a NumPy array (B, H, W, C).
        title (str): Title for the plot.
        max_images (int): Maximum number of images to display in the grid.
    """
    num_images_to_display = min(batch_np.shape[0], max_images)
    
    n_cols = int(np.ceil(np.sqrt(num_images_to_display)))
    n_rows = int(np.ceil(num_images_to_display / n_cols))

    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    plt.suptitle(title, fontsize=16)

    for i in range(num_images_to_display):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        plt.imshow(batch_np[i]) # Directly show the NumPy image
        plt.axis('off')
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


def display_comparison_batch(pred_batch_np: np.ndarray, target_batch_np: np.ndarray, 
                             title: str = "Prediction vs. Target", max_pairs: int = 8):
    """
    Displays pairs of predicted and target images.
    Args:
        pred_batch_np (np.ndarray): Batch of predicted images.
        target_batch_np (np.ndarray): Batch of target (ground truth) images.
        title (str): Title for the overall plot.
        max_pairs (int): Maximum number of image pairs to display.
    """
    num_pairs_to_display = min(pred_batch_np.shape[0], target_batch_np.shape[0], max_pairs)
    
    n_rows = 2
    n_cols = num_pairs_to_display

    plt.figure(figsize=(n_cols * 2.5, n_rows * 2.8))
    plt.suptitle(title, fontsize=16)

    for i in range(num_pairs_to_display):
        ax1 = plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(target_batch_np[i])
        plt.axis('off')

        ax2 = plt.subplot(n_rows, n_cols, n_cols + i + 1)
        plt.imshow(pred_batch_np[i])
        plt.axis('off')
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.98))
    plt.show()


def calculate_pixcorr_and_ssim_single(test_image_path: Path, 
                                      gen_image_path: Path, 
                                      image_size: int = 512) -> tuple[float, float]:
    """
    Calculate pixel correlation between two images.
    
    Args:
        test_image_path: Path to ground truth image
        gen_image_path: Path to generated image  
        image_size: Size to resize images to
        
    Returns:
        Pixel correlation coefficient
    """
    # Load and resize images
    test_image = Image.open(test_image_path).convert('RGB').resize((image_size, image_size))
    gen_image = Image.open(gen_image_path).convert('RGB').resize((image_size, image_size))
    
    # Convert to numpy and normalize to 0-1
    test_image = np.array(test_image) / 255.0
    gen_image = np.array(gen_image) / 255.0
    
    # Flatten and calculate correlation
    target_flat = test_image.reshape(1, -1)
    generated_flat = gen_image.reshape(1, -1)
    
    pixcorr_res = np.corrcoef(target_flat, generated_flat)[0, 1]

    target_gray = rgb2gray(test_image)
    generated_gray = rgb2gray(gen_image)
    ssim_res = ssim(generated_gray, 
                    target_gray, 
                    multichannel=False, 
                    gaussian_weights=True, 
                    sigma=1.5, use_sample_covariance=False, data_range=1.0)
    return pixcorr_res, ssim_res # type: ignore


def calculate_lpips_single(test_image_path: Path,
                           gen_image_path: Path,
                           image_size: int,
                           lpips_model: lpips.LPIPS,
                           device: torch.device) -> float:
    """
    Calculate LPIPS score between two images.
    LPIPS expects images normalized to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])
    lpips_model.to(device)
    
    test_image = Image.open(test_image_path).convert('RGB')
    gen_image = Image.open(gen_image_path).convert('RGB')

    test_tensor = transform(test_image).unsqueeze(0).to(device) # type: ignore
    gen_tensor = transform(gen_image).unsqueeze(0).to(device) # type: ignore

    with torch.no_grad():
        lpips_score = lpips_model(gen_tensor, test_tensor).item()
    return lpips_score


def calculate_fid_score(
    test_image_dir: Path,
    gen_image_dir: Path,
    num_images: int,
    image_size: int,
    device: torch.device
) -> float:
    """
    Calculate Frechet Inception Distance (FID) score.
    FID expects images in [0, 255] range, uint8 or float.
    """
    print(f"Calculating FID score for {num_images} images...")

    # Initialize FID metric
    # feature=2048 corresponds to the default InceptionV3 features
    # normalize=True means input tensors will be normalized to [-1, 1] internally by torchmetrics
    # so we'll load them as [0, 255] float or uint8.
    fid_metric = FID(feature=2048, normalize=True).to(device)

    # Image transformation for FID: resize, to_tensor (0-1 float), then to uint8 (0-255)
    # FID's internal normalize=True will handle the [-1,1] conversion for its InceptionV3.
    transform_fid = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # Converts to [0, 1] float
        lambda x: (x * 255).to(torch.uint8) # Convert to [0, 255] uint8
    ])

    test_images_list = []
    gen_images_list = []

    for i in range(num_images):
        test_path = test_image_dir / f"{i}.png"
        gen_path = gen_image_dir / f"{i}.png"

        if not test_path.exists() or not gen_path.exists():
            print(f"Warning: Image pair {i} not found for FID, skipping.")
            continue

        try:
            test_img = Image.open(test_path).convert('RGB')
            gen_img = Image.open(gen_path).convert('RGB')

            test_images_list.append(transform_fid(test_img))
            gen_images_list.append(transform_fid(gen_img))
        except Exception as e:
            print(f"Error loading image {i} for FID: {e}")
            continue

    if not test_images_list or not gen_images_list:
        print("Not enough valid image pairs to calculate FID.")
        return float('nan')

    # Stack all images into a single tensor for update
    test_tensor_batch = torch.stack(test_images_list).to(device)
    gen_tensor_batch = torch.stack(gen_images_list).to(device)

    # Update FID metric
    fid_metric.update(test_tensor_batch, real=True)
    fid_metric.update(gen_tensor_batch, real=False)

    # Compute FID score
    fid_score = fid_metric.compute().item()
    return fid_score


def calculate_image_level_metrics(
    test_image_dir: Path, 
    gen_image_dir: Path, 
    num_images: int = 200,
    image_size: int = 512,
    lpips_model: lpips.LPIPS = lpips.LPIPS(net='vgg'),
    device: torch.device = torch.device("cuda")
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate pixel correlation, SSIM, and LPIPS for a batch of images.

    Args:
        test_image_dir: Directory containing ground truth images
        gen_image_dir: Directory containing generated images
        num_images: Number of images to evaluate
        image_size: Size to resize images to
        device: Device to run LPIPS on

    Returns:
        Tuple of (pixcorrs, ssims, lpips_scores)
    """
    pixcorrs = []
    ssims = []
    lpips_scores = []
    for i in range(num_images):
        test_path = test_image_dir / f"{i}.png"
        gen_path = gen_image_dir / f"{i}.png"
        
        if not test_path.exists():
            print(f"Warning: Ground truth image {i} not found, skipping")
            continue
            
        if not gen_path.exists():
            print(f"Warning: Generated image {i} not found, skipping")
            continue
        
        try:
            pixcorr_res, ssim_res = calculate_pixcorr_and_ssim_single(test_path, gen_path, image_size)
            pixcorrs.append(pixcorr_res)
            ssims.append(ssim_res)
            lpips_score = calculate_lpips_single(test_path, gen_path, image_size, lpips_model, device)
            lpips_scores.append(lpips_score)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue
    
    return np.array(pixcorrs), np.array(ssims), np.array(lpips_scores)


class FeatureExtractor:
    """Handles feature extraction from various pre-trained models."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.feature_list = []
        
    def _hook_fn(self, module: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        """Hook function to capture intermediate features."""
        self.feature_list.append(outputs.cpu().numpy())
    
    def _setup_model(self, model_name: str, layer: str | int) -> nn.Module:
        """Setup model with appropriate hooks."""
        self.feature_list = []  # Reset feature list
        
        if model_name == 'inceptionv3':
            # model = tvmodels.inception_v3(pretrained=True)
            model = tvmodels.inception_v3(weights=tvmodels.Inception_V3_Weights.DEFAULT)
            if layer == 'avgpool':
                model.avgpool.register_forward_hook(self._hook_fn) # type: ignore
            elif layer == 'lastconv':
                model.Mixed_7c.register_forward_hook(self._hook_fn) # type: ignore
                
        elif model_name == 'alexnet':
            # model = tvmodels.alexnet(pretrained=True)
            model = tvmodels.alexnet(weights=tvmodels.AlexNet_Weights.DEFAULT)
            if layer == 2:
                model.features[4].register_forward_hook(self._hook_fn) # type: ignore
            elif layer == 5:
                model.features[11].register_forward_hook(self._hook_fn) # type: ignore
            elif layer == 7:
                model.classifier[5].register_forward_hook(self._hook_fn) # type: ignore
                
        elif model_name == 'clip':
            clip_model, _ = clip.load("ViT-L/14", device=self.device) # type: ignore
            model = clip_model.visual.to(torch.float32)
            if layer == 7:
                model.transformer.resblocks[7].register_forward_hook(self._hook_fn)
            elif layer == 12:
                model.transformer.resblocks[12].register_forward_hook(self._hook_fn)
            elif layer == 'final':
                model.register_forward_hook(self._hook_fn)
                
        elif model_name == 'efficientnet':
            # model = tvmodels.efficientnet_b1(weights=True)
            model = tvmodels.efficientnet_b1(weights=tvmodels.EfficientNet_B1_Weights.DEFAULT)
            model.avgpool.register_forward_hook(self._hook_fn) # type: ignore
            
        elif model_name == 'swav':
            model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            model.avgpool.register_forward_hook(self._hook_fn) # type: ignore
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.eval() # type: ignore
        model.to(self.device) # type: ignore
        return model # type: ignore
    
    def extract_features(self, model_name: str, layer: str | int, data_loader: DataLoader) -> np.ndarray:
        """Extract features from specified model and layer."""
        print(f"Extracting features from {model_name} layer {layer}")
        
        model = self._setup_model(model_name, layer)
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                _ = model(batch)
        
        # Process features based on model type
        if model_name == 'clip' and layer in [7, 12]:
            features = np.concatenate(self.feature_list, axis=1).transpose((1, 0, 2))
        else:
            features = np.concatenate(self.feature_list)
        
        return features


def calculate_distance_metrics(test_features: np.ndarray, gen_features: np.ndarray, num_images: int) -> float:
    """Calculate correlation distance metrics for specific models."""
    distances = []
    for i in range(num_images):
        dist = scipy.spatial.distance.correlation(test_features[i], gen_features[i])
        distances.append(dist)
    return np.mean(distances) # type: ignore


def calculate_pairwise_correlation(ground_truth: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """
    Calculate pairwise correlation performance metric.
    
    Args:
        ground_truth: Ground truth features
        predictions: Predicted features
        
    Returns:
        Tuple of (performance, p_value)
    """
    r = np.corrcoef(ground_truth, predictions)
    r = r[:len(ground_truth), len(ground_truth):]
    
    # Congruent pairs are on diagonal
    congruents = np.diag(r)
    
    # Count successes (when correlation is lower than congruent)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    
    # Calculate performance
    perf = np.mean(success_cnt) / (len(ground_truth) - 1)
    p = 1 - binom.cdf(
        perf * len(ground_truth) * (len(ground_truth) - 1), 
        len(ground_truth) * (len(ground_truth) - 1), 
        0.5
    )
    
    return perf, p # type: ignore


def extract_and_save_features(
    images_dir: Path, 
    features_dir: Path, 
    model_configs: list, 
    device: torch.device
) -> None:
    """Extract and save features from all specified models."""
    features_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor(device)
    
    for model_config in model_configs:
        model_name = model_config.name
        layers = model_config.layers
        
        if model_name == "clip":
            continue
        for layer in layers:
            print(f"Processing {model_name} layer {layer}")
            
            # Create dataset and dataloader
            dataset = ImageEvaluationDataset(
                images_dir, 
                model_name=model_name,
                num_images=config.evaluation.num_test_images
            )
            dataloader = DataLoader(
                dataset, 
                batch_size=config.evaluation.batch_size, 
                shuffle=False
            )
            
            # Extract features
            features = extractor.extract_features(model_name, layer, dataloader)
            
            # Save features
            save_path = features_dir / f"{model_name}_{layer}.npy"
            np.save(save_path, features)
            print(f"Saved features to {save_path}")


def evaluate_reconstruction_quality(
    test_images_dir: Path,
    gen_images_dir: Path,
    test_features_dir: Path,
    gen_features_dir: Path,
    ranking_metric: str = "composite",
    num_examples_per_category: int = 5,
    device: torch.device = torch.device("cuda")
) -> tuple[dict, dict]:
    """
    Evaluate reconstruction quality using multiple metrics.
    
    Args:
        test_images_dir: Directory containing ground truth images
        gen_images_dir: Directory containing generated images
        test_features_dir: Directory containing ground truth features
        gen_features_dir: Directory containing generated features
        device: Device to run LPIPS and FID on
        
    Returns:
        Dictionary containing all performance metrics
    """
    results = {}
    
    # Calculate image-level metrics
    print("Calculating image-level metrics...")
    pixel_corr, ssim_score, lpips_score = calculate_image_level_metrics(test_images_dir, 
                                                                        gen_images_dir, 
                                                                        config.evaluation.num_test_images,
                                                                        device=device)
    per_image_results = []
    for i in range(len(pixel_corr)):
        composite_score = (pixel_corr[i] + ssim_score[i] + (1 - lpips_score[i])) / 3
        per_image_results.append({
            'index': i,
            'pixcorr': pixel_corr[i],
            'ssim': ssim_score[i],
            'lpips': lpips_score[i],
            'composite': composite_score
        })

    if ranking_metric == 'lpips':
        per_image_results.sort(key=lambda x: x[ranking_metric])
    else:
        per_image_results.sort(key=lambda x: x[ranking_metric], reverse=True)

    n_results = len(per_image_results)
    best_results = per_image_results[:num_examples_per_category]
    worst_results = per_image_results[-num_examples_per_category:]

    middle_start = (n_results - num_examples_per_category) // 2
    medium_results = per_image_results[middle_start:middle_start + num_examples_per_category]
    
    per_image_analysis = {
        'best': best_results,
        'medium': medium_results,
        'worst': worst_results,
        'all_results': per_image_results,
        'ranking_metric': ranking_metric
    }
    
    avg_pixel_corr = np.mean(pixel_corr)
    avg_ssim = np.mean(ssim_score)
    avg_lpips = np.mean(lpips_score)
    results['pixel_correlation'] = avg_pixel_corr
    results['ssim'] = avg_ssim
    results['lpips'] = avg_lpips

    print(f"Pixel Correlation: {avg_pixel_corr:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    
    # Calculate FID score
    print("Calculating FID score...")
    fid_score = calculate_fid_score(
        test_images_dir,
        gen_images_dir,
        num_images=config.evaluation.num_test_images,
        image_size=config.evaluation.image_size,
        device=device
    )
    results['fid'] = fid_score # Add FID to results
    print(f"FID: {fid_score:.4f}")

    # Calculate feature-level metrics
    print("Calculating feature-level metrics...")
    
    for model_config in config.evaluation.models:
        model_name_eval = model_config.name
        layers = model_config.layers
        if model_name_eval == "clip":
            continue
        for layer in layers:
            print(f"Evaluating {model_name_eval} layer {layer}")
            
            # Load features
            test_features_path = test_features_dir / f"{model_name_eval}_{layer}.npy"
            gen_features_path = gen_features_dir / f"{model_name_eval}_{layer}.npy"
            
            if not test_features_path.exists() or not gen_features_path.exists():
                print(f"Warning: Features not found for {model_name_eval}_{layer}, skipping")
                continue
            
            test_features = np.load(test_features_path)
            gen_features = np.load(gen_features_path)
            
            # Reshape features
            test_features = test_features.reshape((len(test_features), -1))
            gen_features = gen_features.reshape((len(gen_features), -1))
            
            # Calculate metrics based on model type
            if model_name_eval in ['efficientnet', 'swav']:
                # Use distance metrics for these models
                distance = calculate_distance_metrics(
                    test_features[:config.evaluation.num_test_images],
                    gen_features[:config.evaluation.num_test_images],
                    config.evaluation.num_test_images
                )
                results[f"{model_name_eval}_{layer}_distance"] = distance
                print(f"Distance: {distance:.4f}")
            else:
                # Use pairwise correlation for other models
                perf, p_value = calculate_pairwise_correlation(
                    test_features[:config.evaluation.num_test_images],
                    gen_features[:config.evaluation.num_test_images]
                )
                results[f"{model_name_eval}_{layer}_pairwise_corr"] = perf
                results[f"{model_name_eval}_{layer}_p_value"] = p_value
                print(f"Pairwise Correlation: {perf:.4f} (p={p_value:.6f})")
    
    return results, per_image_analysis


# def plot_reconstruction_examples(
#     test_images_dir: Path,
#     gen_images_dir: Path,
#     per_image_analysis: dict[str, list[dict]],
#     image_size: int = 512,
#     save_path: Path | None = None
# ) -> None:
#     """
#     Plot best, medium, and worst reconstruction examples.
    
#     Args:
#         test_images_dir: Directory containing ground truth images
#         gen_images_dir: Directory containing generated images
#         per_image_analysis: Results from evaluate_reconstruction_quality with per-image metrics
#         image_size: Size to resize images to
#         save_path: Optional path to save the plot
#     """
#     best_results = per_image_analysis['best']
#     medium_results = per_image_analysis['medium']
#     worst_results = per_image_analysis['worst']
#     ranking_metric = per_image_analysis['ranking_metric']
    
#     num_examples_per_category = len(best_results)
    
#     # Create visualization
#     fig, axes = plt.subplots(3, num_examples_per_category * 2, 
#                            figsize=(num_examples_per_category * 6, 12))
    
#     categories = [
#         ("Best", best_results),
#         ("Medium", medium_results),
#         ("Worst", worst_results)
#     ]
    
#     for cat_idx, (category_name, category_results) in enumerate(categories):
#         for ex_idx, result in enumerate(category_results):
#             img_idx = result['index']
            
#             # Load images
#             test_path = test_images_dir / f"{img_idx}.png"
#             gen_path = gen_images_dir / f"{img_idx}.png"
            
#             test_img = Image.open(test_path).convert('RGB').resize((image_size, image_size))
#             gen_img = Image.open(gen_path).convert('RGB').resize((image_size, image_size))
            
#             # Convert to arrays
#             test_array = np.array(test_img)
#             gen_array = np.array(gen_img)
            
#             # Plot ground truth
#             ax_gt = axes[cat_idx, ex_idx * 2]
#             ax_gt.imshow(test_array)
#             ax_gt.set_title(f"{category_name} - Ground Truth\n(Image {img_idx})")
#             ax_gt.axis('off')
            
#             # Plot generated
#             ax_gen = axes[cat_idx, ex_idx * 2 + 1]
#             ax_gen.imshow(gen_array)
#             title = f"{category_name} - Generated\n"
#             title += f"{ranking_metric}: {result[ranking_metric]:.3f}"
#             ax_gen.set_title(title)
#             ax_gen.axis('off')
    
#     plt.tight_layout()
#     plt.suptitle(f"Reconstruction Quality Analysis (Ranked by {ranking_metric})", 
#                  fontsize=16, y=0.98)
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")
    
#     plt.show()


def plot_reconstruction_examples(
    test_images_dir: Path,
    gen_images_dir: Path,
    per_image_analysis: dict[str, list[dict]],
    image_size: int = 512,
    save_path: Path | None = None,
    random_selection: bool = False,
    num_random_pairs: int = 8,
    seed: int | None = None
) -> None:
    """
    Plot best, medium, and worst reconstruction examples, or random pairs.
    
    Args:
        test_images_dir: Directory containing ground truth images
        gen_images_dir: Directory containing generated images
        per_image_analysis: Results from evaluate_reconstruction_quality with per-image metrics
        image_size: Size to resize images to
        save_path: Optional path to save the plot
        random_selection: If True, plot random pairs instead of ranked examples
        num_random_pairs: Number of random pairs to plot (when random_selection=True)
        seed: Random seed for reproducibility
    """
    if random_selection:
        # Generate random selection
        if seed is not None:
            np.random.seed(seed)
        
        # Find available indices
        available_indices = []
        for i in range(config.evaluation.num_test_images):
            test_path = test_images_dir / f"{i}.png"
            gen_path = gen_images_dir / f"{i}.png"
            if test_path.exists() and gen_path.exists():
                available_indices.append(i)
        
        # Select random indices
        selected_indices = np.random.choice(available_indices, size=min(num_random_pairs, len(available_indices)), replace=False)
        
        # Create fake analysis structure for random selection
        random_results = [{'index': idx, 'composite': 0.0} for idx in selected_indices]
        categories = [("Random", random_results)]
        ranking_metric = "random"
        
        # Calculate grid dimensions: 5 pairs per row
        pairs_per_row = 5
        n_rows = (len(random_results) + pairs_per_row - 1) // pairs_per_row
        
        # Create visualization with proper grid
        fig, axes = plt.subplots(n_rows, pairs_per_row * 2, 
                               figsize=(pairs_per_row * 6, n_rows * 4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
    else:
        # Use existing ranked analysis
        best_results = per_image_analysis['best']
        medium_results = per_image_analysis['medium']
        worst_results = per_image_analysis['worst']
        ranking_metric = per_image_analysis['ranking_metric']
        num_examples_per_category = len(best_results)
        
        categories = [
            ("Best", best_results),
            ("Medium", medium_results),
            ("Worst", worst_results)
        ]
        
        # Create visualization
        fig, axes = plt.subplots(3, num_examples_per_category * 2, 
                               figsize=(num_examples_per_category * 6, 12))
    
    for cat_idx, (category_name, category_results) in enumerate(categories):
        for ex_idx, result in enumerate(category_results):
            img_idx = result['index']
            
            if random_selection:
                # Calculate row and column for random layout
                row = ex_idx // 5
                col = ex_idx % 5
                ax_gt = axes[row, col * 2]
                ax_gen = axes[row, col * 2 + 1]
            else:
                # Use original layout for ranked examples
                ax_gt = axes[cat_idx, ex_idx * 2]
                ax_gen = axes[cat_idx, ex_idx * 2 + 1]
            
            # Load images
            test_path = test_images_dir / f"{img_idx}.png"
            gen_path = gen_images_dir / f"{img_idx}.png"
            
            test_img = Image.open(test_path).convert('RGB').resize((image_size, image_size))
            gen_img = Image.open(gen_path).convert('RGB').resize((image_size, image_size))
            
            # Convert to arrays
            test_array = np.array(test_img)
            gen_array = np.array(gen_img)
            
            # Plot ground truth
            ax_gt.imshow(test_array)
            ax_gt.set_title(f"{category_name} - Ground Truth\n(Image {img_idx})")
            ax_gt.axis('off')
            
            # Plot generated
            ax_gen.imshow(gen_array)
            title = f"{category_name} - Generated\n"
            if not random_selection:
                title += f"{ranking_metric}: {result[ranking_metric]:.3f}"
            ax_gen.set_title(title)
            ax_gen.axis('off')
    
    # Hide empty subplots for random selection
    if random_selection:
        total_pairs = len(selected_indices)
        pairs_per_row = 5
        n_rows = (total_pairs + pairs_per_row - 1) // pairs_per_row
        for i in range(total_pairs, n_rows * pairs_per_row):
            row = i // pairs_per_row
            col = i % pairs_per_row
            axes[row, col * 2].set_visible(False)
            axes[row, col * 2 + 1].set_visible(False)
    
    plt.tight_layout()
    title = f"Random Reconstruction Pairs (n={len(selected_indices)})" if random_selection else f"Reconstruction Quality Analysis (Ranked by {ranking_metric})"
    plt.suptitle(title, fontsize=16, y=0.98)

    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()