import rclpy
import cv2
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge

class DepthImageNode(Node):
    def __init__(self):
            super().__init__('Depth_node')
            self.frame = 0
            self.cam_pub = self.create_publisher(Image, '/depth_pub_image', 10)
            self.cam_sub = self.create_subscription(Image, '/depth_sub_image', self.subscribe_callback,  10)
            self.cap = cv2.VideoCapture(0)
            self.br = CvBridge()
            self.timer = self.create_timer(0.1, self.publish_video)  # 每 0.1 秒呼叫一次

    def publish_video(self):
        ret, frame = self.cap.read()
          
        if ret == True:
       
            self.cam_pub.publish(self.br.cv2_to_imgmsg(frame, encoding = 'bgr8'))
    
            # Display the message on the console
            self.get_logger().info('Publishing video frame')

    def subscribe_callback(self, data):
        self.get_logger().info('Receiving video frame')
 
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data, encoding = 'bgr8')
        
        # Display image
        cv2.imshow("camera", current_frame)
        
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageNode()
    rclpy.spin(node)                                
    rclpy.shutdown()

if __name__ == "__main__":
    main()




