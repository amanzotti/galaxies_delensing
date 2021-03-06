from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys
import Code
here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.txt', 'CHANGES.txt')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='glaxies_delensing',
    # version=glaxies_delensing.__version__,
    url='https://github.com/amanzotti/galaxies_delensing',
    license='Apache Software License',
    author='AM',
    # install_requires=['Flask>=0.10.1',
    #                   'Flask-SQLAlchemy>=1.0',
    #                   'SQLAlchemy==0.8.2',
    #                   ],
    # cmdclass={'test': PyTest},
    # author_email='jeff@jeffknupp.com',
    description='Test CMB delensing with galaxies',
    # long_description=long_description,
    packages=['galaxies_delensing'],
    include_package_data=True,
    platforms='any',
    # test_suite='sandman.test.test_sandman',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',

    ],
    extras_require={
        'testing': ['pytest'],
    }
)
