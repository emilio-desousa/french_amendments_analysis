import amendements_analysis.settings.base as stg
import wget
import os
import zipfile

def download_data():
    zip_file_path = os.path.join(stg.RAW_DATA_DIR,stg.ZIP_FILE_NAME)
    if os.path.isfile(zip_file_path):
        print('Zip File already here')
    else:
        print('Downloading...')
        wget.download(
                stg.URL_TO_DL_DATA,
                zip_file_path
            )
        print('Done!')
    print('Extracting...')  
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(stg.RAW_DATA_DIR)
    print('Done!')

