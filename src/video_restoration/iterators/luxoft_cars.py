import os
import cv2
import numpy as np


class LuxoftCars(object):

    def __init__(self, data_path, skip=20, stack=10):
        self.data_path = data_path
        self.skip = skip
        self.stack = stack

    def iter_items(self):
        video_dirs = os.listdir(self.data_path)
        for up_dir in video_dirs:
            # Create path and dir
            in_path_dir = os.path.join(self.data_path, up_dir)

            # Search inside
            in_files = os.listdir(in_path_dir)
            for in_f in in_files:
                video_frames_path = os.path.join(in_path_dir, in_f)
                video_frames = os.listdir(video_frames_path)
                video_frames = sorted(video_frames, key=lambda x: int(x.split('.')[0]))

                frames_stack = []
                for fr_idx, fr_path in enumerate(video_frames):
                    frame_img = cv2.cvtColor(cv2.imread(os.path.join(video_frames_path, fr_path)), cv2.COLOR_BGR2RGB)
                    if fr_idx % self.skip == 0:
                        frames_stack.append(frame_img)
                        if len(frames_stack) == self.stack:
                            frames_stack_f = frames_stack
                            frames_stack = frames_stack[1:]
                            yield np.stack(frames_stack_f, axis=-1)

    def __iter__(self):
        return self.iter_items()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from video_restoration.utils.tools import SequentialBatchCollector
    lc = LuxoftCars(data_path="/home/filip141/Datasets/Cars_Luxoft-Images", skip=2, stack=10)
    sbc = SequentialBatchCollector(lc)
    batch_x, batch_y = sbc.collect_batch()
    for img in lc:
        plt.imshow(img[..., -1])
        plt.show()
