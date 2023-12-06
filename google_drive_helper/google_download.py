import os, io, sys, argparse
from Google import Create_Service
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIENT_SECRET_FILE = f'{root_dir}/credentials.json'
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

def drive_download(
    files = [{'id': "1kjsebdHAVTcqPhnU_owLky8gLhbDPvxr", 'name': 'test.jpg'}],
    download_folder = 'results',
):
    for file in tqdm(files):
        done = False
        while not done:
            try:
                file_id, file_path = file['id'], file['name']
                file_path = os.path.join(root_dir, download_folder, file_path)

                if os.path.exists(file_path):
                    done = True
                    break

                header = os.path.dirname(file_path)
                if not os.path.exists(header):
                    os.makedirs(header)

                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fd=fh, request=request)
                done = False
                
                while not done:
                    _, done = downloader.next_chunk()
                
                fh.seek(0)
                with open(file_path, "wb") as f:
                    f.write(fh.read())
                    f.close()
                done = True
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                sys.exit(0)
            except Exception:
                print("Unexpected error:", sys.exc_info()[0])
                print(f"Error downloading file: {file_path}. Retrying...")
            

def list_dir(
    directory_id = '1UwQv5bJdL9LJOQyNWXyfdx6CyDM94z9v',
    directory_name = 'emu_results',
):
    all_files = []
    page_token = None

    # List all folders in the specified directory
    while True:
        folder_results = service.files().list(
            q=f"'{directory_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            fields='nextPageToken, files(id, name)',
            pageToken=page_token,
        ).execute()
        folders = folder_results.get('files', [])

        for folder in folders:
            print(f"Processing folder: {folder['name']}")
            file_folder_results = list_dir(folder['id'], folder['name'])
            for file in file_folder_results:
                file['name'] = directory_name + '/' + file['name']
                all_files.append(file)

        page_token = folder_results.get('nextPageToken', None)
        if page_token is None: break            

    # list all files in the specified directory
    while True:
        file_results = service.files().list(
            q=f"'{directory_id}' in parents and mimeType!='application/vnd.google-apps.folder'",
            fields='nextPageToken, files(id, name)',
            pageToken=page_token,
        ).execute()
        files = file_results.get('files', [])

        for file in files:
            file['name'] = directory_name + '/' + file['name']
            all_files.append(file)

        page_token = file_results.get('nextPageToken', None)
        if page_token is None: break

    return all_files

def main(
    dirs = [
        {'id': '1NERWpRJHjmcuQ-mIHhsmCPRuA7XNutuN', 'name': 'emu_results'},
        {'id': '1qthlfg6wiIoi6sq0lVaBvg407z10h__J', 'name': 'gill_results'},
        {'id': '1bTvMU5HOprhAWAW-wNZmaN5hTmRFZiSu', 'name': 'seed_results'}
    ],
    download_folder = 'results',
):

    for dir in dirs:
        print('Starting to download files from', dir['name'])
        all_files = list_dir(dir['id'], dir['name'])
        drive_download(all_files, download_folder=download_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from Google Drive')
    parser.add_argument('--id', type=str, default=None, help='ID of the file to download')
    parser.add_argument('--name', type=str, default=None, help='Name of the file to download')
    parser.add_argument('--download_folder', type=str, default='results', help='Folder to download files to')
    args = parser.parse_args()
    
    if args.id is None or args.name is None:
        main(download_folder=args.download_folder)
    else:
        main(
            [{
                'id': args.id,
                'name': args.name
            }],
            download_folder=args.download_folder,
        )
    
    