from setuptools import setup, find_packages


def get_version() -> str:
    return "2.0.0"


print(find_packages())

setup(
    name="kinodata",
    version=get_version(),
    packages=find_packages(),
)
