import rclpy
from rclpy.node import Node
from robot_msgs.msg import LinePointsArray  # type: ignore

class LineSubscriberNode(Node):
    def __init__(self):
        super().__init__('line_subscriber3')
        self.sub = self.create_subscription(
            LinePointsArray,
            'candidates',
            self.line_callback,
            10
        )

    def line_callback(self, msg: LinePointsArray):
        if not msg.points:
            return

        for idx, point in enumerate(msg.points):
            self.get_logger().info(
                f'[{idx}] cx={point.cx}, cy={point.cy}, lost={point.lost}'
            )


def main():
    rclpy.init()
    node = LineSubscriberNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
