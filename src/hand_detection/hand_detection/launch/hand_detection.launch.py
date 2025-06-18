from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1) Camera node
    cam = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        parameters=[
            {'video_device': '/dev/video2'},
            {'io_method': 'read'},
            {'image_size': [640, 480]},
            {'framerate': 30},
            {'output_encoding': 'rgb8'},
        ],
        output='screen'
    )

    # 2) Hand-detection node
    hand_detector = Node(
        package='hand_detection',
        executable='hand_detection_node',
        name='hand_detection_node',
        output='screen'
    )

    # 3) Viewer for the annotated image
    hand_viewer = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='hand_image_viewer',
        # Remap the viewer’s "image" topic to your detection output
        remappings=[('image', '/hand_detection/image_annotated')],
        output='screen'
    )

    return LaunchDescription([
        cam,
        hand_detector,
        hand_viewer
    ])
