import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download():
    # Set your Kaggle API credentials (make sure to upload your kaggle.json file to your notebook)
    os.environ["KAGGLE_USERNAME"] = "ajfunari"
    os.environ["KAGGLE_KEY"] = "c39847e9465527cf36d02a2ed041042d"

    # # Create a Kaggle API object
    api = KaggleApi()
    api.authenticate()

    # # Specify the dataset you want to download
    dataset_name = "tawsifurrahman/covid19-radiography-database"

    # # Set the destination folder for the downloaded dataset
    current_directory = os.getcwd()
    download_path = "/input"
    path = current_directory + download_path
    print(path)

    # # Download the dataset
    api.dataset_download_files(dataset_name, path=path, unzip=True)  # 800 MB

if __name__ == "__main__":
    download()