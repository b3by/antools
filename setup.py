import codecs
import os

from setuptools import setup
from setuptools import find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


setup(
    name='antools',
    packages=find_packages(exclude=['tests']),
    version='0.0.1',
    description='Boilerplating for data science projects',
    long_description_content_type='text/markdown',
    long_description=read('README.md'),
    author='Antonio Bevilacqua',
    author_email='b3by.in.th3.sky@gmail.com',
    url='https://github.com/b3by/boilerscience',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
    ],
    python_requires='>=3',
    install_requires=[
        'tensorflow',
        'coverage'
    ],
    include_package_data=True,
    keywords='datascience boilerplate'
)
