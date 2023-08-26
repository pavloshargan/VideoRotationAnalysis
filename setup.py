from setuptools import setup, find_packages

setup(
    name="videorotation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "tensorflow"
    ],
    package_data={
    'videorotation': ['rotnet_street_view_resnet50_keras2_pb/*', 'rotnet_street_view_resnet50_keras2_pb/variables/*'],
    },
    author="Pavlo Sharhan",
    author_email="activetechinnovators@example.com",
    description="A module to analyze video rotation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pavloshargan/VideoRotationAnalysis",
)