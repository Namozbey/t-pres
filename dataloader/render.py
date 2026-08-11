import argparse
from dataloader.utils import save_sketches, setup_dataset
from config import TRAINING_CONFIG

def main():
    # 1. Initialize the argument parser
    parser = argparse.ArgumentParser(description="Pipeline to download 3D meshes and render RGB/Sketch pairs.")
    
    # 2. Define the command-line flags with your requested defaults
    parser.add_argument(
        '--category', 
        type=str, 
        default=TRAINING_CONFIG["category"],
        help='The Objaverse/LVIS category to process (default: chair)'
    )
    parser.add_argument(
        '--download_limit', 
        type=int, 
        default=3,
        help='Maximum number of 3D meshes to download (default: 3)'
    )
    parser.add_argument(
        '--num_views', 
        type=int, 
        default=5,
        help='Number of camera views to render per mesh (default: 6)'
    )

    # 3. Parse the arguments provided by the user
    args = parser.parse_args()

    # Print a status header
    print("=" * 50)
    print(f"🚀 STARTING RENDER PIPELINE")
    print(f"Category:       {args.category}")
    print(f"Download Limit: {args.download_limit} meshes")
    print(f"Camera Views:   {args.num_views} per mesh")
    print("=" * 50)

    # 4. Execute the pipeline
    try:
        print("\n[1/2] Setting up dataset...")
        # setup_dataset(download_limit=args.download_limit, category=args.category)
        
        print("\n[2/2] Generating and saving sketches...")
        save_sketches(num_views=args.num_views, category=args.category)
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()