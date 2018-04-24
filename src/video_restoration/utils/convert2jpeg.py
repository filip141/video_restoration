import os
import cv2
import shutil
import numpy as np
import matplotlib.image as mpimg

from video_restoration.utils.ffmpeg_tools import FFMPEGVideoReader


class Video2JpegConverter(object):

    def __init__(self, path="", new_path="", ffmpeg_path="/usr/bin/ffmpeg", resize="128x128"):
        self.path = path
        self.new_path = new_path
        self.ffmpeg = ffmpeg_path
        self.resize = [int(x) for x in resize.split("x")]
        if os.path.isdir(self.new_path):
            shutil.rmtree(self.new_path)

    def process(self):
        os.mkdir(self.new_path)
        video_dirs = os.listdir(self.path)
        for up_dir in video_dirs:
            # Create path and dir
            old_path_dir = os.path.join(self.path, up_dir)
            new_path_dir = os.path.join(self.new_path, up_dir)
            os.mkdir(new_path_dir)

            # Search inside
            in_files = os.listdir(old_path_dir)
            for in_f in in_files:
                video_file = os.path.join(old_path_dir, in_f)
                video_images_dir = os.path.join(new_path_dir, in_f.split(".")[0])

                # Create FFMPEG instance
                try:
                    ffmpeg = FFMPEGVideoReader(self.ffmpeg, video_file, resolution=None)
                except AttributeError:
                    print("{} is not video file!".format(video_file))
                    continue

                # Create folder in new dir
                os.mkdir(video_images_dir)

                # Create instance in batch
                im_idx = 0
                while True:
                    frame = ffmpeg.next_frame()
                    if frame is None:
                        break
                    frame = cv2.resize(frame, tuple(self.resize), interpolation=cv2.INTER_AREA)
                    im_path = os.path.join(video_images_dir, "{}.png".format(im_idx))
                    print("Creating: {}".format(im_path))
                    mpimg.imsave(im_path, frame)
                    im_idx += 1
                ffmpeg.kill()


if __name__ == '__main__':
    v2j = Video2JpegConverter("/home/filip141/Datasets/Cars_Luxoft", "/home/filip141/Datasets/Cars_Luxoft-Images")
    v2j.process()