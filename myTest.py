import kagglehub

# Download latest version
path = kagglehub.dataset_download("muqaddasejaz/fake-reviews-dataset")

print("Path to dataset files:", path)