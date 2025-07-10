import kagglehub

# Download latest version
path = kagglehub.dataset_download("dataclusterlabs/suitcaseluggage-dataset")

print("Path to dataset files:", path)
