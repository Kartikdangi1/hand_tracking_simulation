from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Package paths
    pkg_share = get_package_share_path('dexhand_description')
    xacro_file = pkg_share / 'urdf' / 'dexhand-right.xacro'
    rviz_cfg   = pkg_share / 'rviz' / 'urdf.rviz'
    ctrl_yaml  = pkg_share / 'config' / 'controllers.yaml'

    # Launch arguments
    gui_arg   = DeclareLaunchArgument(
        'gui', default_value='true', choices=['true', 'false'],
        description='Enable joint_state_publisher_gui')
    model_arg = DeclareLaunchArgument(
        'model', default_value=str(xacro_file),
        description='Absolute path to robot xacro')
    rviz_arg  = DeclareLaunchArgument(
        'rvizconfig', default_value=str(rviz_cfg),
        description='Absolute path to rviz config')

    # Robot description parameter (XACRO -> URDF)
    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('model')]),
        value_type=str
    )

    """    # 1) ros2_control: Load URDF + controllers
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description},
            str(ctrl_yaml)
        ],
        output='screen'
    )

    # 2) Spawn trajectory controller
    spawn_traj_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'dexhand_traj_controller',
            '--controller-manager', '/controller_manager'
        ],
        output='screen'
    )
    """
    # 3) Joint state publishers
    jsp = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        condition=UnlessCondition(LaunchConfiguration('gui'))
    )
    jsp_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    # 4) Robot state publisher & RViz
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen'
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
        output='screen'
    )

    # 5) Vision and mapping nodes
    cam_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='v4l2_camera_node',
        parameters=[
            {'video_device': '/dev/video2'},
            {'io_method': 'read'},
            {'image_size': [640, 480]},
            {'framerate': 30},
            {'output_encoding': 'rgb8'}
        ],
        output='screen'
    )
    hand_det = Node(
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
    map_node = Node(
        package='hand_detection',
        executable='hand2dexhand_node',
        name='hand2dexhand',
        output='screen',
            parameters=[{
        'publish_topic': '/dexhand_joint_poses',
        'rate':          30.0,
        'frame_id':      'base_link',  # or 'camera' if you prefer
        'mirrored':      False          # True for left-hand input
    }]
    )

    return LaunchDescription([
        gui_arg,
        model_arg,
        rviz_arg,
        #ros2_control_node,
        #spawn_traj_controller,
        jsp,
        jsp_gui,
        rsp,
        rviz,
        cam_node,
        hand_det,
        hand_viewer,
        map_node,
    ])
