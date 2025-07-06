import argparse
from evaluation_utils import extract_and_save_features, evaluate_reconstruction_quality, plot_reconstruction_examples
from configs.config import config
from pathlib import Path
import torch


def main():
    args = parse_args()
    sub_id = args.sub_id
    model_name = args.model_name
    extract_features = args.extract_features
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_images_dir = Path(f"{config.data.image_dir}test_images_direct/")
    test_features_dir = Path("cache") / "test_features"
    gen_images_dir = Path(f"results/unclip/{model_name}/sub-{sub_id:02d}/")
    gen_features_dir = Path("cache") / "gen_features"

    if extract_features:
        print("Extracting test features...")
        extract_and_save_features(test_images_dir, test_features_dir, config.evaluation.models, device)
        print("Extracting generated features...")
        extract_and_save_features(gen_images_dir, gen_features_dir, config.evaluation.models, device)
    
    print("Evaluating reconstruction quality...")
    results, per_image_analysis = evaluate_reconstruction_quality(test_images_dir, 
                                                                  gen_images_dir, 
                                                                  test_features_dir, 
                                                                  gen_features_dir,
                                                                  ranking_metric="lpips",
                                                                  device=device)
    plot_reconstruction_examples(
        test_images_dir, gen_images_dir, per_image_analysis,
        save_path=Path("reconstruction_examples.png")
    )

    plot_reconstruction_examples(
        test_images_dir, gen_images_dir, per_image_analysis,
        save_path=Path("reconstruction_examples_random.png"),
        random_selection=True,
        num_random_pairs=15,
        seed=42
    )


    print("\n=== EVALUATION SUMMARY ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate neural decoder reconstruction quality")
    parser.add_argument("--sub_id", type=int, required=True, help="Subject ID")
    parser.add_argument("--model_name", type=str, required=True, choices=["ridge", "mlp"])
    parser.add_argument("--extract_features", action="store_true", 
                       help="Whether to extract features (set if features don't exist)")
    return parser.parse_args()


if __name__ == "__main__":
    main()