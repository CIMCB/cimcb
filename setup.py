from setuptools import setup


def readme():
  with open('README.md', encoding='utf-8') as f:
    return f.read()


setup(
    name="cimcb",
    version="2.1.2",
    description="A package containing the necessary tools for the statistical analysis of untargeted and targeted metabolomics data.",
    long_description=readme(),
    long_description_content_type='text/markdown',
    license="MIT",
    url="https://github.com/KevinMMendez/cimcb",
    packages=["cimcb", "cimcb.bootstrap", "cimcb.cross_val", "cimcb.model", "cimcb.plot", "cimcb.utils"],
    python_requires=">=3.5",
    install_requires=["bokeh>=1.0.0",
                      "keras>=2.2.4",
                      "numpy>=1.12",
                      "pandas",
                      "scipy",
                      "scikit-learn",
                      "statsmodels",
                      "theano",
                      "tqdm",
                      "xlrd",
                      "joblib"],
    author="Kevin Mendez, David Broadhurst",
    author_email="k.mendez@ecu.edu.au, d.broadhurst@ecu.edu.au",
)
