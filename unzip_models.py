# importing the zipfile module
import shutil
from zipfile import ZipFile
import os
import warnings


def unzip(zip_path, out_path, model_name):
    # for every zip file in the folder
    for path in os.listdir(zip_path):
        if path.endswith(".zip"):
            save_name = path.split(".")[0]
            new_folder_name = model_name + "-" + save_name

            if os.path.exists(out_path + new_folder_name):
                warnings.warn(f"Folder '{new_folder_name}' already exists. Skipping...")
            else:
                print("Decompressing: " + path)

                with ZipFile(zip_path + path, 'r') as zipObj:

                    # Extract all the contents of zip file in current directory
                    zipObj.extractall(out_path)

                    print("Renamed to: " + new_folder_name)
                    # rename the folder to fit the model name
                    os.rename(out_path + save_name, out_path + new_folder_name)

                    print("Save name: " + path)

                # remove the auto generated __MACOSX folder
                shutil.rmtree(out_path + "__MACOSX")

if __name__ == '__main__':

    zip_path = "./zip_models/"
    out_path = "./models/"
    model_name = "xlm-mlm-17-1280-finetuned-ner"

    unzip(zip_path, out_path, model_name)
