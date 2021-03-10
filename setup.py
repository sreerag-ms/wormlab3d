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
        'numpy >= 1.19, <1.20'
    ],
    extras_require={
        'test': [
            'pytest'
        ],
        'inv': [
            'torch >= 1.7, <= 1.8',
            'matplotlib >= 3.3',
            'scikit-learn >= 0.24',
            'tensorboard == 2.4.1',
        ]
    },
    python_requires='>=3.9, <3.10'
)
