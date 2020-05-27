import os
from pathlib import Path

def download_youtube_video(youtube_id, youtube_format, youtube_file):
    try:
        import youtube_dl
    except:
        print('installing youtube_dl')
        os.system('pip install youtube-dl')
        import youtube_dl

    if not Path(youtube_file).exists():
        youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'
        print(f'downloading {youtube_url}')
        os.system(f'youtube-dl {youtube_url} -f {youtube_format} -o video.mp4')
    else:
        print(f'youtube video already present')


def download_google_drive_file(file_id, file_path):
    import gdown
    if not Path(file_path).exists():
        print('downloading model weights')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', file_path, quiet=False)


def install_detectron2_colab():
    try:
        import detectron2
        return True
    except:
        print('installing detectron2')
    import subprocess
    import sys

    def pip(args):
        subprocess.check_call([sys.executable, "-m", "pip"] + args)

    commands = """
!pip install DOESNT-EXIST
!pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

!pip uninstall -y opencv-python
!pip uninstall -y opencv-contrib-python
!pip install opencv-python==3.4.2.16
!pip install opencv-contrib-python==3.4.2.16
!pip install opencv-contrib-python-nonfree

!pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
    """
    for c in (c for c in commands.split('\n') if len(c) > 0):
        header = '!pip '
        if c.startswith(header):
            args = c[len(header):].split(' ')
            pip(args)


if __name__ == '__main__':
    print('testing functions')
    install_detectron2_colab()
