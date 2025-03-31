from huggingface_hub import HfApi

repo_id = "ezruby/scale-equity-nlp"
api = HfApi()

api.upload_folder(
    folder_path="/scratch/ez2545/scale-equity-nlp/Data",
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"Upload complete: {repo_id}")
