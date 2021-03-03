from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open("version", 'r') as f:
    version = f.read()

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

reqs = [
    "numpy==1.20.1",
    "pandas==1.2.2",
    "python-dateutil==2.8.1",
    "pytz==2021.1",
    "six==1.15.0",
]

test_reqs = ['pytest~=5.4.1']

setup(
    name='ff3',
    version=version,
    description="Fast & Furious 3",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://ff3.io/",
    license="Apache 2.0",
    packages=pkgs,
    classifiers=["License :: OSI Approved :: Apache Software License",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Information Technology",
                 "Topic :: Software Development :: Libraries :: Python Modules"],
    install_requires=reqs,
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    test_suite='tests',
    tests_require=test_reqs
)
