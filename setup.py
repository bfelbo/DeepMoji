from setuptools import setup

setup(
    name='deepmoji',
    version='1.0',
    packages=['deepmoji'],
    description='DeepMoji library',
    include_package_data=True,
    install_requires=[
        'emoji==0.4.5',
        'h5py==2.7.0',
        'Keras==2.13.1',
        'numpy==1.13.1',
        'scikit-learn==0.19.0',
        'text-unidecode==1.0',
    ],
)
