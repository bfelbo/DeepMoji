from setuptools import setup

setup(
    name='deepmoji',
    version='1.0',
    packages=['deepmoji'],
    description='DeepMoji library',
    long_description=open("README.md", encoding="UTF-8").read(),
    long_descrition_content_type='text/markdown',
    license="MIT",
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
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
