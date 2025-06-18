from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hand_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
       # Install all Python launch files
       (os.path.join('share', package_name, 'launch'),
        glob(os.path.join('hand_detection', 'launch', '*.launch.py'))),
           (os.path.join('share', package_name, 'config'),
        glob(os.path.join( 'config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='idmp',
    maintainer_email='usama.ali@thws.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_detection_node = hand_detection.hand_detection_node:main',
            'hand2dexhand_node = hand_detection.hand2dexhand_node:main',
        ],
    },
)
