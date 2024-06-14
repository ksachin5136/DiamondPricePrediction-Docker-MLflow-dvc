from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Sachin K',
    author_email='ksachin5136@gmail.com',

    # specify the dependencies that need to be installed for your Python package(ie. your local package)
    # install_requires=["scikit-learn","pandas","numpy"],
    install_requires=get_requirements('requirements.txt'),

    # When you use find_packages() in your setup.py script,
    # it automatically finds and lists all the Python packages and sub-packages in your project directory.(local packages)
    # When you run python setup.py install or pip install ., the setup.py script will install all these discovered packages.
    # with this when you import the local packages somewhere in your code then it fund it else it will give Error as ;
    # ModuleNotFoundError: No module named 'src'

    # so there are Three ways to install  local packages
    # 1. pip install .
    # 2. python setup.py install
    # 3. In requirements.txt in the last write -e.  this '-e' specify an editable install, which means even if you update your
    # local packages and write more classes or methods ,
    # those class and methods will be availabl to components  without any re-install of local packages


    # packages=find_packages(where="src")
    packages=find_packages()
)
