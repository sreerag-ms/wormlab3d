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
        'matplotlib >= 3.3, < 3.4',
        'mongoengine >= 0.22, < 0.23',
        'numpy >= 1.20, < 1.21',
        'opencv >= 4.5, < 4.6',
        'torch >= 1.7, <= 1.8',
        'pims >= 0.5, <= 0.6',
        'pymongo >= 3.11, < 3.12',
        'python-blosc >= 1.10, < 1.11',
        'python-dotenv >= 0.16, < 1',
        'pytorch >= 1.8, < 1.9',
        'scikit-image >= 0.18, < 0.19',
        'scikit-learn >= 0.24, < 0.25',
        'tensorboard >= 2.4, < 2.5',
        'torchvision >= 0.9, < 1.0'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
    },
    python_requires='>=3.9, <3.10'
)
