from setuptools import setup
from setuptools import find_packages

setup(
    name='antools',
    package=find_packages(exclude=['tests']),
    version='0.0.1',
    description='Boilerplating for data science projects',
    author='Antonio Bevilacqua',
    author_email='b3by.in.th3.sky@gmail.com',
    url='https://github.com/b3by/boilerscience',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3',
    install_requires=[
        'tensorflow',
        'coverage'
    ],
    keywords='datascience boilerplate'
)
