import glob
import os

import pandas as pd


def _load_accumulated_info(root_folder_path,
                           dataset_folder_name="PZA confidential/crops",
                           image_folder_name="crops"):
    dataset_folder_path = os.path.join(root_folder_path, dataset_folder_name)
    image_folder_path = os.path.join(dataset_folder_path, image_folder_name)

    accumulated_info_list = []
    for subdir in os.listdir(image_folder_path):
        subdir_path = os.path.join(image_folder_path, subdir)
        image_file_paths = glob.glob(os.path.join(subdir_path, "*.jpg"))
        splits = subdir.split('_')
        identity_id = splits[1]
        class_id = splits[2]

        if int(class_id) == 0:
            # TODO: better scheme to select image
            mid = int(len(image_file_paths) / 2)
            for image_file_path in image_file_paths[mid-2:mid+2]:
                camera_id = int(os.path.basename(image_folder_path)[:4])

                # Append the records
                accumulated_info = {
                    "image_file_path": image_file_path,
                    "identity_ID": identity_id,
                    "camera_ID": camera_id
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
        image_folder_name="2008-GroteMarkt-Q3-RichtingMaalderijstraat-2020-08-17-07h59min58s623ms")
    test_query_accumulated_info_dataframe = _load_accumulated_info(
        root_folder_path=root_folder_path,
        image_folder_name="2009-GroteMarkt-Q4-RichtingStadhuis-2020-08-17-07h59min59s773ms")

    print(train_and_valid_accumulated_info_dataframe)
    print(test_query_accumulated_info_dataframe)
    print(test_gallery_accumulated_info_dataframe)

    return (train_and_valid_accumulated_info_dataframe,
            test_query_accumulated_info_dataframe,
            test_gallery_accumulated_info_dataframe)
