import os
import cv2
import copy
import time
import random
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt


class FFMPEGVideoReader(object):

    def __init__(self, ffmpeg_bin, video_path, resolution="1920x1080"):
        self.ffmpeg_bin = ffmpeg_bin
        self.video_path = video_path
        devnull = open(os.devnull, "w")

        # Automatically find resolution
        if resolution is None:
            info_cmd = [self.ffmpeg_bin,
                        '-i', self.video_path]
            ffmpeg_pipe = sp.Popen(info_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            stdout, stderr = ffmpeg_pipe.communicate()
            try:
                resolution = str(stderr).split("Stream")[1].split(",")[3].rstrip().lstrip()
            except IndexError:
                raise AttributeError("File is not Video file or it can't be processed by FFMPEG.")
        self.resolution = [int(x) for x in resolution.split("x")]
        self.command = [self.ffmpeg_bin,
                        '-i', self.video_path,
                        '-f', 'image2pipe',
                        '-pix_fmt', 'rgb24',
                        '-vcodec', 'rawvideo', '-']
        self.ffmpeg_pipe = self.open_ffmpeg_pipe(devnull)

    def open_ffmpeg_pipe(self, devnull):
        try:
            ffmpeg_pipe = sp.Popen(self.command, stdout=sp.PIPE, bufsize=10**8, stderr=devnull)
        except OSError:
            time.sleep(1)
            ffmpeg_pipe = self.open_ffmpeg_pipe(devnull)
        return ffmpeg_pipe

    def length(self, fps):
        command = [self.ffmpeg_bin,
                   '-i', self.video_path]
        result = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
        duration = [x.decode("utf-8").split(",")[0].lstrip() for x in result.stdout.readlines()
                    if "Duration" in str(x)][0].split(": ")[1].split(":")
        duration = int(duration[0]) * 3600 + int(duration[1]) * 60 + float(duration[2])
        return int(duration * fps)

    def seek(self, number):
        self.ffmpeg_pipe.stdout.read(np.prod(self.resolution) * 3 * number)

    def next_frame(self):
        # Read bytes from ffmpeg pipe
        raw_image = self.ffmpeg_pipe.stdout.read(np.prod(self.resolution) * 3)

        # Convert to uint8 and reshape image
        image = np.fromstring(raw_image, dtype='uint8')
        if image.size == 0:
            return None
        try:
            image = image.reshape((self.resolution[1], self.resolution[0], 3))
        except ValueError:
            return None
        self.ffmpeg_pipe.stdout.flush()
        return image

    def iter_items(self):
        img = self.next_frame()
        while img is not None:
            img = self.next_frame()
            yield img

    def __iter__(self):
        return self.iter_items()

    def kill(self):
        self.ffmpeg_pipe.kill()


class FFMPEGVideoWritter(object):

    def __init__(self, ffmpeg_bin, video_path, resolution="1920x1080"):
        self.ffmpeg_bin = ffmpeg_bin
        self.video_path = video_path
        self.resolution = [int(x) for x in resolution.split("x")]
        self.command = [self.ffmpeg_bin, '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-s', resolution,
                        '-pix_fmt', 'rgb24',
                        '-r', '24',  '-i', '-',
                        '-an',
                        '-vcodec', 'mpeg4',
                        video_path]
        self.ffmpeg_pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)

    def save_frame(self, data):
        self.ffmpeg_pipe.stdin.write(data.tobytes())

    def kill(self):
        self.ffmpeg_pipe.stdin.close()
        self.ffmpeg_pipe.stderr.close()
        self.ffmpeg_pipe.kill()


if __name__ == '__main__':
    ffmpeg_bin = "/usr/bin/ffmpeg"
    video_path = "/home/filip141/Datasets/Cars_Luxoft/Cars_Luxoft/2017_10_21_13_17_27.3gp"
    ffmpeg_video = FFMPEGVideoReader(ffmpeg_bin, video_path, resolution=None)
    plt.imshow(ffmpeg_video.next_frame())
    plt.show()
