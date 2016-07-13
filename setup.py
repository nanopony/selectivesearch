import os
from setuptools import setup, Extension

# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name="selectivesearch",
        version="0.0.1",
        author="Nanopony",
        author_email="sddeath@gmail.com",
        description=("A python-wrapper for selective search algorithm implemented by (c) 2014 The MatConvNet team"),
        license="MIT",
        keywords="selectivesearch opencv image segmentation",
        url="http://github.com/nanopony/selectivesearch",
        packages=['selectivesearch'],
        install_requires=['numpy'],
        ext_modules=[
            Extension('selectivesearch._selectivesearch',
                      ['selectivesearch/_selectivesearch.cpp',
                       'selectivesearch/selectivesearch.cpp'],
                      include_dirs=['selectivesearch'],
                      library_dirs=['/'],
                      libraries=['opencv_core'],
                      extra_compile_args=['-g']
                      )
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
            "License :: OSI Approved :: MIT License",
        ],
)
