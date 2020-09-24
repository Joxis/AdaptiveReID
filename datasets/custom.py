import glob
import os

import pandas as pd


def _load_accumulated_info(root_folder_path,
                           dataset_folder_name="custom",
                           image_folder_name="images"):
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)
    image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    image_file_path_list = sorted(
        glob.glob(os.path.join(image_folder_path, "*.jpg")))

    accumulated_info_list = []
    for image_file_path in image_file_path_list:
        # Extract identity_ID
        image_file_name = image_file_path.split(os.sep)[-1]
        identity_ID = int(image_file_name.split("_")[0])

        # Extract camera_ID
        cam_seq_ID = image_file_name.split("_")[1]
        camera_ID = int(cam_seq_ID[1])

        # Append the records
        accumulated_info = {
            "image_file_path": image_file_path,
            "identity_ID": identity_ID,
            "camera_ID": camera_ID
        }
        accumulated_info_list.append(accumulated_info)

    # Convert list to data frame
    accumulated_info_dataframe = pd.DataFrame(accumulated_info_list)
    return accumulated_info_dataframe


def load_custom(root_folder_path):
    train_and_valid_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="2005-GroteMarkt-2020-08-17-07h59min58s787ms")
    test_gallery_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="2006-GroteMarkt-Q1-RichtingWisselstraat-2020-08-17-07h59min55s767ms")
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path, image_folder_name="2007-GroteMarkt-Q2-RichtingKaasrui-2020-08-17-07h59min58s353ms")

    print(test_query_accumulated_info_dataframe)

    return train_and_valid_accumulated_info_dataframe, test_query_accumulated_info_dataframe, test_gallery_accumulated_info_dataframe
