import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from mocap4r2_msgs.msg import RigidBodies
from stable_baselines3 import PPO
import numpy as np
import math
from pathlib import Path


class PPOPublisherNode(Node):
    def __init__(self):
        super().__init__('ppo_publisher_node')

        default_model = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "standard"
            / "ppo_standard.zip"
        )
        self.declare_parameter("model_path", str(default_model))
        self.declare_parameter("spider_body_name", "10")
        self.declare_parameter("target_body_name", "9")
        self.declare_parameter("command_topic", "cm550_command")
        self.declare_parameter("mocap_topic", "rigid_bodies")
        self.declare_parameter("control_period", 1.0)
        self.declare_parameter("success_radius", 0.2)

        self.spider_body_name = self.get_parameter("spider_body_name").value
        self.target_body_name = self.get_parameter("target_body_name").value
        command_topic = self.get_parameter("command_topic").value
        mocap_topic = self.get_parameter("mocap_topic").value
        self.success_radius = float(self.get_parameter("success_radius").value)

        self.spider_body = None
        self.target_body = None
        self.steps = 0

        self.publisher_ = self.create_publisher(Int32, command_topic, 10)
        self.rigid_body_suscriber = self.create_subscription(
            RigidBodies,
            mocap_topic,
            self.positions_callback,
            10
        )

        self.get_logger().info("Cargando modelo PPO...")
        model_path = self.get_parameter("model_path").value
        self.policy = PPO.load(model_path, device='cpu')
        if self.policy.observation_space.shape != (2,):
            raise ValueError(
                "El nodo de referencia requiere una política con observación (2,). "
                f"El modelo declara {self.policy.observation_space.shape}."
            )
        self.get_logger().info("Modelo cargado. Esperando datos del Mocap...")


        self.commands = {
            0: 227, 1: 228, 2: 251, 3: 252, 4: 253,
            5: 256, 6: 258, 7: 262, 8: 266, 9: 267, 10: 261, 11: 263
        }

        self.timer_period = float(self.get_parameter("control_period").value)
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.init_pose_sent = False
        self.goal_reached = False
    
    def get_heading_from_quaternion(self, q):
        """
        Calcula el Ángulo de Giro (Heading) alrededor del eje vertical (Y) de Optitrack.
        
        q: Cuaternión del mensaje de posición.
        Retorna: Ángulo de giro en radianes.
        """
        # Esta es la fórmula para la rotación alrededor del eje Y (Pitch en convención Z-Up)
        # que usamos como Heading en tu sistema Y-Up.
        t3 = +2.0 * (q.w * q.y - q.z * q.x)
        t4 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        heading_y = math.atan2(t3, t4)
        return heading_y

    def transform_to_local_frame(self, target_pose, robot_pose):
        """
        Transforma la posición del Target (X_global, Z_global) al marco local (X'_fwd, Y'_lat) del robot.
        - Optitrack Global: Z=Forward, X=Lateral, Y=Vertical.
        - Policy Local: X'=Forward, Y'=Lateral.
        """
        
        # 1. Obtener posiciones globales y remapear según la convención Optitrack (Z=Forward, X=Lateral)
        rx_lat = robot_pose.position.x # Componente Lateral (X global)
        rz_fwd = robot_pose.position.z # Componente Forward (Z global)
        tx_lat = target_pose.position.x
        tz_fwd = target_pose.position.z

        # 2. Obtener orientación del robot (Heading alrededor de Y)
        robot_heading = self.get_heading_from_quaternion(robot_pose.orientation)
        #self.get_logger().info(f"ROBOT HEADING: {robot_heading}")

        # 3. Traslación
        dx_lat = tx_lat - rx_lat # Delta Lateral
        dz_fwd = tz_fwd - rz_fwd # Delta Forward
       # self.get_logger().info(f"Deltas (traslación): {dx_lat}, {dz_fwd}")

        # 4. Rotación 2D Inversa (Llevar el vector [dz_fwd, dx_lat] al marco local)
        # El vector de observación es [Forward, Lateral].
        cos_h = math.cos(robot_heading)
        sin_h = math.sin(robot_heading)
        
        # Local X' (Forward Component)
        # X' = Fwd_delta * cos(h) + Lat_delta * sin(h)
        local_z_forward = dz_fwd * cos_h + dx_lat * sin_h
        
        # Local Y' (Lateral Component)
        # Y' = -Fwd_delta * sin(h) + Lat_delta * cos(h)
        local_x_lateral = -dz_fwd * sin_h + dx_lat * cos_h

        return local_z_forward, local_x_lateral

    def send_command(self, value: int):
        msg = Int32()
        msg.data = value
        self.publisher_.publish(msg)

    def positions_callback(self, msg: RigidBodies):
        for body in msg.rigidbodies:
            if body.rigid_body_name == self.spider_body_name:
                self.spider_body = body
            elif body.rigid_body_name == self.target_body_name:
                self.target_body = body

    def control_loop(self):
        if self.target_body is None or self.spider_body is None:
            self.get_logger().warn("Esperando Mocap...", throttle_duration_sec=2)
            return

        if not self.init_pose_sent:
            self.get_logger().info("Estableciendo pose inicial (Stand up)!")
            self.send_command(201)
            self.init_pose_sent = True
            return 

        try:
            # 1. Obtener observación local
            local_forward, local_lateral = self.transform_to_local_frame(
                self.target_body.pose,
                self.spider_body.pose
            )

            distance = math.hypot(local_forward, local_lateral)
            if distance <= self.success_radius:
                if not self.goal_reached:
                    self.get_logger().info("Objetivo alcanzado; enviando pose de reposo.")
                    self.send_command(201)
                    self.goal_reached = True
                return
            self.goal_reached = False

            # 2. Observación para la policy: [forward, lateral].
            obs = np.array([local_forward, local_lateral], dtype=np.float32)

            # 3. Inferencia
            action, _ = self.policy.predict(obs, deterministic=True)
            command = self.commands[int(action)]
            self.steps += 1
            
            self.get_logger().info(
                f"Local Obs: fwd={local_forward:.2f}, lat={local_lateral:.2f} | "
                f"Cmd: {command}"
            )
            
            self.send_command(command)

        except Exception as e:
            self.get_logger().error(f"Error en cálculos de transformación: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PPOPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
