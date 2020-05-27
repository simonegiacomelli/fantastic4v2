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


