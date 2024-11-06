from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''this function will return a list of all the requirements'''
    requirements=[]
    hypen_e_dot='-e .'
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(
    name='Online_payment_fraud_detection',
    version='0.0.1',
    author='Dayaban Sagar',
    author_email='dayabansgr@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)