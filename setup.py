from setuptools import setup, find_packages

setup(
    name="videorotation",
    version='0.2',
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "tensorflow"
    ],
    package_data={
    'videorotation': ['mobilenet2_bi-rotnet.tflite'],
    },
    author="Pavlo Sharhan",
    author_email="activetechinnovators@example.com",
    description="A module that allows to estimate whether video is flipped",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pavloshargan/VideoRotationAnalysis",
)