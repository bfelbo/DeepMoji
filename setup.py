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
        'emoji>=0.4.5,<1.0.0',
        'h5py>=2.7.0,<3.0.0',
        'Keras>=2.3.1,<3.0.0',
        'numpy>=1.18.1,<2.0.0',
        'scikit-learn>=0.19.0,<1.0.0',
        'text-unidecode>=1.0,<2.0',
    ],
    tests_require=[
        "nose>=1.3.7,<2.0.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
)
