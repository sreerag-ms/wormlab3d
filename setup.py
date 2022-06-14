from setuptools import setup

setup(
    name='wormlab3d',
    version='0.0.1',
    description='Worms in 3D.',
    author='Tom Ilett, Tom Ranner',
    url='https://gitlab.com/tom0/wormlab3d',
    packages=['wormlab3d'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'av >= 8.0, < 8.1',
        'fastcluster >= 1.2, < 1.3',
        'ffmpeg-python >= 0.2, < 0.3',
        'flask >= 2.0, < 2.1',
        'flask_wtf >= 1.0.0, < 1.1',
        'gunicorn >= 20.1, < 21',
        'h5py >= 3.2, < 3.3',
        'mayavi >= 4.7, < 4.8',
        'matplotlib >= 3.4, < 3.5',
        'mongoengine >= 0.22, < 0.23',
        'numpy >= 1.20, < 1.21',
        'opencv-python >= 4.5, < 4.6',
        'pims >= 0.5, <= 0.6',
        'progress >= 1.6, < 2.0',
        'pymongo >= 3.11, < 3.12',
        'blosc >= 1.10, < 1.11',
        'python-dotenv >= 0.16, < 1',
        'torch >= 1.10, < 1.11',
        'scikit-image >= 0.18, < 0.19',
        'scikit-learn >= 0.24, < 0.25',
        'scipy >= 1.8, < 1.9',
        'simple-worm[inv] @ git+https://gitlab.com/tom-ranner/simple-worm.git@master',
        'tensorboard >= 2.4, < 2.5',
        'torchvision >= 0.11, < 1.0'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
    },
    python_requires='>=3.9, <3.10'
)
