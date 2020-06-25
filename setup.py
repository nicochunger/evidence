import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evidence",
    version="0.1.4",
    author="Nicolas Unger",
    author_email="nicolas.unger@unige.ch",
    description="Calculates the bayesian evidence for radial velocity models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicochunger/evidence",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    package_data={'': ['*.so']}
)
