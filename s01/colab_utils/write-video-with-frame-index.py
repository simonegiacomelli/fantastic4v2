import s01.colab_utils as u
import cv2

youtube_file = 'video.mp4'
u.download_youtube_video('JZdqjtWsL0U', '"best[height=360]"', youtube_file)

video = cv2.VideoCapture(youtube_file)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

output_file = cv2.VideoWriter(
    filename='video-output.mkv',
    # some installation of opencv may not support x264 (due to its license),
    # you can try other format (e.g. MPEG o XVID o x264)
    fourcc=cv2.VideoWriter_fourcc(*"x264"),
    fps=frames_per_second,
    frameSize=(width, height),
    isColor=True,
)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (width - 120, height - 10)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

for idx, frame in enumerate(u.video_frames(youtube_file, 0, None, apply_COLOR_RGB2BGR=False)):
    lbl = f'{idx}'.rjust(4, '0')
    cv2.putText(frame, lbl,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    output_file.write(frame)

output_file.release()
