import os
import sys
import glob
import platform
import subprocess
import argparse
import pandas as pd

def setup_environment():
    """Sets the PyTorch attention backend based on the OS."""
    env = os.environ.copy()
    
    if platform.system() == "Windows":
        print("🪟 Windows Detected: Forcing xformers and native spconv[cite: 1].")
        env.pop("SSL_CERT_FILE", None)
        env["ATTN_BACKEND"] = "xformers"
        env["SPCONV_ALGO"] = "native"
    else:
        print("🐧 Linux Detected: Using high-performance flash-attn.")
        env.pop("ATTN_BACKEND", None) 
        
    return env

def run_command(command, env):
    """Executes terminal commands and halts on failure."""
    print(f"\n▶️ Running: {' '.join(command)}")
    result = subprocess.run(command, env=env)
    
    if result.returncode != 0:
        print(f"\n❌ Pipeline stopped. Error in step: {command[1]}")
        sys.exit(1)

def generate_metadata(data_dir):
    """Creates the base metadata.csv by scanning existing .glb meshes."""
    print(f"\n[📝] Scanning {data_dir}/meshes/ to generate metadata.csv...")
    
    mesh_dir = os.path.join(data_dir, "meshes")
    mesh_files = glob.glob(os.path.join(mesh_dir, "*.glb"))
    
    if not mesh_files:
        print(f"⚠️ No .glb meshes found in {mesh_dir}!")
        sys.exit(1)
        
    records = []
    for mesh_path in mesh_files:
        uid = os.path.basename(mesh_path).split('.')[0]
        records.append({
            "id": uid,                                # Fixed: Added ID
            "sha256": uid,                            
            "local_path": f"meshes/{uid}.glb",        # Fixed: Added relative local path
            "rendered": False,
            "voxelized": False,
            "ss_encoded": False,
            "feature_dinov2_vitl14_reg": False,
            "encoded": False
        })
        
    df = pd.DataFrame(records)
    
    # Ensure columns are in the exact order as your original CSV
    columns_order = ['id', 'sha256', 'local_path', 'rendered', 'voxelized', 'ss_encoded', 'feature_dinov2_vitl14_reg', 'encoded']
    df = df[columns_order]
    
    csv_path = os.path.join(data_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Initial metadata.csv created for {len(df)} objects.")
    return csv_path

def update_csv_flag(csv_path, column_name, value=True):
    """Updates status columns in metadata to unlock downstream scripts[cite: 2]."""
    df = pd.read_csv(csv_path)
    df[column_name] = value
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV Updated: {column_name} = {value}")

def main():
    parser = argparse.ArgumentParser(description="Full TRELLIS Data Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='Path (e.g., dataloader/data/chair)')
    parser.add_argument('--dataset_name', type=str, default='Toys4k', help='Internal dataset name')
    
    args = parser.parse_args()
    OUTPUT_DIR = args.data_dir
    env = setup_environment()
    
    # 1. Initialize Metadata based on existing meshes
    csv_path = generate_metadata(OUTPUT_DIR)
    
    # 2. TRELLIS Rendering (Produces multi-view RGBs needed for voxelization)
    run_command(["python", "TRELLIS/dataset_toolkits/render.py", args.dataset_name, "--output_dir", OUTPUT_DIR], env)
    update_csv_flag(csv_path, "rendered", True)
    
    # 3. Voxelization
    run_command(["python", "TRELLIS/dataset_toolkits/voxelize.py", args.dataset_name, "--output_dir", OUTPUT_DIR], env)
    update_csv_flag(csv_path, "voxelized", True)
    
    # 4. Feature Extraction (DINOv2)
    update_csv_flag(csv_path, "feature_dinov2_vitl14_reg", False)
    run_command(["python", "TRELLIS/dataset_toolkits/extract_feature.py", "--output_dir", OUTPUT_DIR], env)
    update_csv_flag(csv_path, "feature_dinov2_vitl14_reg", True)
    
    # 5. SS Latent Encoding
    run_command(["python", "TRELLIS/dataset_toolkits/encode_ss_latent.py", "--output_dir", OUTPUT_DIR], env)
    update_csv_flag(csv_path, "ss_encoded", True)
    
    # 6. Structured Latent Encoding (The final SLAT files)
    run_command(["python", "TRELLIS/dataset_toolkits/encode_latent.py", "--output_dir", OUTPUT_DIR], env)
    update_csv_flag(csv_path, "encoded", True)
    
    print(f"\n🎉 SLAT generation complete! Latents ready in {os.path.join(OUTPUT_DIR, 'latents')}.")

if __name__ == "__main__":
    main()