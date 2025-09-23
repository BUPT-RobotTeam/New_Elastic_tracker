#!/usr/bin/env python3
import select
import sys
import termios
import tty
import rospy
from std_msgs.msg import Int16
from std_msgs.msg import Float32MultiArray


class ActorKeyboard(object):
    def __init__(self):
        rospy.init_node("actor_keyboard")

        self.forward_speed = 0
        self.turn_speed = 0

        self.test_pub_1 = rospy.Publisher("forward", Int16, queue_size=3)
        self.test_pub_2 = rospy.Publisher("turn", Int16, queue_size=3)
        self.cmd_pub = rospy.Publisher("/actor_cmd", Float32MultiArray, queue_size=3)

        # 定时器：100ms发布一次指令（原逻辑不变）
        self.pub_timer = rospy.Timer(rospy.Duration(nsecs=100000000), self.pub_key)
        # 超时定时器：500ms检查一次（原逻辑不变）
        self.timeout_timer = rospy.Timer(rospy.Duration(nsecs=500000000), self.check_timeout)
        self.last_forward_control_time = None
        self.last_turn_control_time = None

        # 新增：记录终端原始设置，确保退出时恢复（避免终端卡死）
        self.original_termios = termios.tcgetattr(sys.stdin)

    def check_timeout(self, time_event):
        # 原逻辑不变：超时后重置速度
        if self.last_forward_control_time:
            if (rospy.Time.now() - self.last_forward_control_time).to_sec() > 0.2:
                self.forward_speed = 0
        if self.last_turn_control_time:
            if (rospy.Time.now() - self.last_turn_control_time).to_sec() > 0.2:
                self.turn_speed = 0

    def get_key(self, key_timeout=0.1):  # 关键修复1：超时时间设为0.1s（避免忙等待）
        try:
            # 设置终端为原始模式（监听键盘）
            tty.setraw(sys.stdin.fileno())
            # select超时0.1s：每0.1s检查一次键盘输入，空闲时释放CPU（解决卡死）
            rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
            if rlist:
                key = sys.stdin.read(1)
                # 键盘控制逻辑（原逻辑不变）
                if key == "i":
                    self.forward_speed = 1
                    self.last_forward_control_time = rospy.Time.now()
                elif key == "j":
                    self.turn_speed = -1
                    self.last_turn_control_time = rospy.Time.now()
                elif key == "l":
                    self.turn_speed = 1
                    self.last_turn_control_time = rospy.Time.now()
                elif key == "u":
                    self.forward_speed = 1
                    self.turn_speed = -1
                    self.last_forward_control_time = rospy.Time.now()
                    self.last_turn_control_time = rospy.Time.now()
                elif key == "o":
                    self.forward_speed = 1
                    self.turn_speed = 1
                    self.last_forward_control_time = rospy.Time.now()
                    self.last_turn_control_time = rospy.Time.now()
                elif key == "k":
                    self.forward_speed = 0
                    self.turn_speed = 0
                    self.last_forward_control_time = rospy.Time.now()
                    self.last_turn_control_time = rospy.Time.now()
                elif key == "q":  # 按q直接正常退出
                    self.restore_terminal()  # 恢复终端设置
                    rospy.signal_shutdown("User pressed 'q'")
                    sys.exit(0)
            else:
                key = ''
            return key
        finally:
            # 关键修复2：无论是否有输入，都恢复终端设置（避免终端卡死）
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_termios)

    def pub_key(self, time_event):
        # 原逻辑不变：发布速度指令
        self.test_pub_1.publish(self.forward_speed)
        self.test_pub_2.publish(self.turn_speed)
        data = Float32MultiArray(data=[float(self.forward_speed), float(self.turn_speed)])  # 确保是float类型
        self.cmd_pub.publish(data)

    def restore_terminal(self):
        # 恢复终端原始设置（避免退出后终端无法正常输入）
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_termios)

    def run(self):
        try:
            # 关键修复3：用rospy.is_shutdown()判断节点状态，支持Ctrl+C中断
            while not rospy.is_shutdown():
                self.get_key()  # 调用修复后的get_key
        except rospy.ROSInterruptException:
            # 捕获Ctrl+C触发的中断信号
            rospy.loginfo("Actor keyboard node interrupted by Ctrl+C")
        finally:
            # 无论何种退出方式，都恢复终端设置
            self.restore_terminal()
            rospy.loginfo("Actor keyboard node exited, terminal restored")


if __name__ == '__main__':
    print("草泥马")
    try:
        actor_keyboard = ActorKeyboard()
        actor_keyboard.run()
    except Exception as e:
        # 捕获其他异常，确保终端恢复
        rospy.logerr(f"Actor keyboard node error: {str(e)}")
        # 若已初始化终端设置，恢复它
        if 'actor_keyboard' in locals() and hasattr(actor_keyboard, 'original_termios'):
            actor_keyboard.restore_terminal()