import rclpy
import cv2
import torch
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
from PIL import Image as PILImage
from torchvision.models.segmentation import (deeplabv3_resnet101,  DeepLabV3_ResNet101_Weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DepthImageNode(Node):
    def __init__(self):
            super().__init__('Depth_node')
            self.frame = 0
            self.cam_pub = self.create_publisher(Image, '/depth_pub_image', 10)
            self.cam_sub = self.create_subscription(Image, '/depth_pub_image', self.subscribe_callback,  10)
            self.seg_pub = self.create_publisher(Image, '/seg_pub_image', 10)
            self.cap = cv2.VideoCapture(0)
            self.cvbr = CvBridge()
            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            self.model = deeplabv3_resnet101(weights = weights)
            self.model.eval()
            self.transforms = weights.transforms()
            self.timer = self.create_timer(0.1, self.publish_video)  # 每 0.1 秒呼叫一次
            self.model.to(device)

    def publish_video(self):
        ret, frame = self.cap.read()
          
        if ret == True:
            imgmsg = self.cv_inference(frame)
            self.cam_pub.publish(self.cvbr.cv2_to_imgmsg(frame, encoding = 'bgr8'))
            self.seg_pub.publish(imgmsg)
            # Display the message on the console
            self.get_logger().info('Publishing video frame')

    def subscribe_callback(self, data):
        self.get_logger().info('Receiving video frame')
        # Convert ROS Image message to OpenCV image
        current_frame = self.cvbr.imgmsg_to_cv2(data, desired_encoding = 'bgr8')
        # Display image
        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)

    def cv_inference(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(img)
        img = self.transforms(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.to(device)
        with torch.no_grad():
            output = self.model(img)['out'][0]   #['out']：取出主要輸出張量
        output_predictions = output.argmax(0)
        output_predictions = output_predictions.byte().cpu().numpy()
        output_predictions = cv2.applyColorMap(output_predictions, cv2.COLORMAP_JET)
        output_predictions = cv2.cvtColor(output_predictions, cv2.COLOR_RGB2BGR)
        img_tomsg = self.cvbr.cv2_to_imgmsg(output_predictions, encoding='bgr8')
        return img_tomsg


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageNode()
    rclpy.spin(node)                                
    rclpy.shutdown()

if __name__ == "__main__":
    main()




