from setuptools import setup

setup(
    name='deepmoji',
    version='1.0',
    packages=['deepmoji'],
    description='DeepMoji library',
    include_package_data=True,
    install_requires=[
        'emoji',
        'h5py',
        'Keras',
        'numpy',
        "tensorflow",
        'scikit-learn',
        'text-unidecode',
    ],
    tests_require=[
        "nose",
    ]
)
