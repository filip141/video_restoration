from distutils.core import setup

setup(
    name='video_restoration',
    version='0.1',
    packages=['video_restoration', 'video_restoration.utils'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='filip141',
    author_email='filip141@gmail.com',
    description='', requires=['numpy', 'cv2', 'tensorflow_gpu', 'scipy', 'matplotlib', 'tensorflow']
)

