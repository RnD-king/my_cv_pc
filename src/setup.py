from setuptools import find_packages, setup
import glob
import os

package_name = 'my_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.*'))),
        ('share/' + package_name + '/config', glob.glob(os.path.join('config', '*.yaml*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noh',
    maintainer_email='noh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ball_and_hoop = my_cv.ball_and_hoop:main',
            'ball_detect = my_cv.ball_detect:main',
            'ball_recieve = my_cv.ball_recieve:main',
            'color_mask_test = my_cv.color_mask_test:main',
            'depth_test = my_cv.depth_test:main',
            'hoop_detect = my_cv.hoop_detect:main',
            'hurdle_detect = my_cv.hurdle_detect:main',
            'image_saver_roi_keypress = my_cv.image_saver_roi_keypress:main',
            'line_publisher = my_cv.line_publisher:main',
            'line_subscriber = my_cv.line_subscriber:main',
            'line_tracker = my_cv.line_tracker:main',
            'realsense_test = my_cv.realsense_test:main',
        ],
    },
)
