#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'scikit-learn',  'dataclasses', 'hdbscan']

test_requirements = ['pytest>=3', ]

setup(
    author="Maximillian Weil",
    author_email='maximillian.f.weil@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="A Python package to identify, cluster and track modes calculated during an Operational Modal Analysis. ",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='oma_tracking',
    name='oma_tracking',
    packages=find_packages(include=['oma_tracking', 'oma_tracking.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/WEILMAX/oma_tracking',
    version='0.1.0',
    zip_safe=False,
)
