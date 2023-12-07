import os, sys
from Google import Create_Service
from googleapiclient.http import MediaFileUpload

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIENT_SECRET_FILE = f'{root_dir}/credentials.json'
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

def drive_upload(
    files = [{'mime_type': 'application/json', 'path': f"{root_dir}/results/clip_evaluation/[1]colortocolorobject_emu_1shot_mean(25.1)(19.56)(27.49)(0.51)(0.03).json"}],
    upload_folder = '1OrDj-2dcy4-QV0MRdHalBFA8GASnXDD2',
    overwrite = 1,
):
    file_ids = []
    for file in files:
        done = False
        file['name'] = os.path.basename(file['path'])

        old_file_id = None
        if overwrite:
            # Query to search for files with the specified name
            query = f"name = '{file['name']}' and parents = '{upload_folder}'"
            response = service.files().list(
                q=query, 
                spaces='drive', 
                fields='files(id, name, trashed)'
            ).execute()
            # delete the files with the same name and choose one file to overwrite
            old_files = response.get('files', [])
            for old_file in old_files:
                if old_file_id:
                    response = service.files().delete(fileId=old_file.get('id')).execute()
                else:
                    if not old_file.get('trashed', False): # Check if 'trashed' is False
                        old_file_id = old_file.get('id')

        while not done:
            try:
                file_metadata = {
                    'name': file['name'],
                    'parents': [upload_folder]
                }
                media = MediaFileUpload(
                    file['path'], 
                    mimetype=file['mime_type'],
                    resumable = True,
                )

                if overwrite and old_file_id:
                    response = service.files().update(
                        fileId=old_file_id,
                        media_body=media,
                        fields='id'
                    ).execute()
                    print(f"File {file['name']} updated successfully, file id {response.get('id')}.")
                else:
                    response = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id'
                    ).execute()
                    print(f"File {file['name']} uploaded successfully, file id {response.get('id')}.")
                file_ids.append(response.get('id'))
                done = True
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                sys.exit(0)
            except Exception:
                continue

    return file_ids