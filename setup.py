from setuptools import setup

setup(
    name='deepmoji',
    version='2.0',
    packages=['deepmoji'],
    description='DeepMoji library',
    include_package_data=True,
    install_requires=[
        'emoji>=0.4.5,<1.0.0',
        'h5py>=2.7.0,<3.0.0',
        'numpy>=1.18.1,<2.0.0',
        'scikit-learn>=0.19.0,<1.0.0',
        'text-unidecode>=1.0,<2.0',
    ],
    tests_require=[
        "nose>=1.3.7,<2.0.0",
    ],
    extras_require={
        "tensorflow_backend": [
            "tensorflow>=2.0.0,!=2.1.0,<3.0.0",  # tensorflow==2.1.0 has import issues on windows
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
