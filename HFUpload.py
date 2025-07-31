from huggingface_hub import upload_folder, create_repo

# Configuration
repo_id = "ValerianFourel/SOCmappingRastersAndSoilSamples"
local_folder = "./SOCmappingData"  # Your folder containing the zip

def upload_soc_data():
    """Upload SOCmappingData folder to Hugging Face"""

    try:
        # Create the repository (skip if exists)
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"✅ Repository created/confirmed: {repo_id}")

        # Upload the entire folder
        print("🚀 Uploading SOCmappingData folder...")
        upload_folder(
            folder_path=local_folder,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload SOC mapping dataset"
        )

        print("✅ Upload complete!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_soc_data()
