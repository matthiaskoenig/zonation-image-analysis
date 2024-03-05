from zia.annotations.data_sets.create_jena_data import create_jena_dataset
from zia.annotations.data_sets.create_sample_data import create_sample_dataset
from zia.annotations.data_sets.upload_predictions import prepare_and_upload

if __name__ == "__main__":
    create_sample_dataset()
    create_jena_dataset()

    prepare_and_upload(["sample_data", "jena_data"])


