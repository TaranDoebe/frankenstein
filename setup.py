from setuptools import setup, find_packages

setup(
    name='fmri_classification_project',
    version='0.1.0',
    author='Your Name/Team',
    author_email='your.email@example.com',
    description='A project for fMRI-based classification.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/yourrepository', # Optional
    packages=find_packages(exclude=("tests*",)),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'nibabel',
        'tqdm',
        'PyYAML',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8', # Specify your Python version
    entry_points={
        'console_scripts': [
            'run_fmri_classification=main:main',
        ],
    },
)