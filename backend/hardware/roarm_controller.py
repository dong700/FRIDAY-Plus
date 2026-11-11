"""
RoArm-M3 机械臂控制器

支持两种通信方式：
1. HTTP API (WiFi) - 通过 HTTP GET 请求发送 JSON 指令
2. Serial (串口) - 通过 USB/UART 串口通信

JSON指令参考: https://www.waveshare.net/wiki/RoArm-M3-S_JSON%E6%8C%87%E4%BB%A4%E5%90%AB%E4%B9%89
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import get_logger

logger = get_logger("hardware.roarm")


# ============ 关节定义 ============

class Joint(Enum):
    """机械臂关节枚举"""
    BASE = 1        # 底座关节 (0 ~ +-3.14)
    SHOULDER = 2    # 肩关节 (0 ~ +-1.57)
    ELBOW = 3       # 肘关节 (1.57 ~ 3.14)
    WRIST = 4       # 手腕关节1 (0 ~ +-1.57)
    ROLL = 5        # 手腕关节2 / Roll (0 ~ +-3.14)
    GRIPPER = 6     # 夹爪/末端关节 (3.14 ~ 1.08, 减小=张开)


@dataclass
class JointState:
    """单个关节状态"""
    joint: Joint
    radian: float       # 弧度值
    angle: float        # 角度值
    load: int           # 负载
    torque_on: bool     # 扭矩锁是否开启


@dataclass
class ArmState:
    """机械臂完整状态"""
    x: float           # X坐标 mm
    y: float           # Y坐标 mm
    z: float           # Z坐标 mm
    pitch: float       # T轴角度(俯仰) rad
    base_rad: float    # 底座弧度
    shoulder_rad: float # 肩弧度
    elbow_rad: float   # 肘弧度
    wrist_rad: float   # 手腕弧度
    roll_rad: float    # Roll弧度
    gripper_rad: float # 夹爪弧度
    voltage: float     # 电压 V
    joints: List[JointState]
    timestamp: str


@dataclass
class RoArmConfig:
    """机械臂连接配置"""
    mode: str = "http"              # "http" 或 "serial"
    http_ip: str = "192.168.4.1"    # HTTP模式IP
    http_port: int = 80             # HTTP端口
    serial_port: str = ""            # 串口名如 COM3
    serial_baudrate: int = 115200   # 波特率
    timeout: float = 5.0            # 请求超时秒数


class RoArmController:
    """
    RoArm-M3 机械臂控制器

    封装HTTP和串口双通信方式，提供高级控制接口。

    使用方法:
        controller = RoArmController(RoArmConfig(mode="http", http_ip="192.168.4.1"))
        await controller.connect()

        # 移动到坐标
        await controller.move_to_xyz(x=200, y=50, z=200)

        # 控制夹爪
        await controller.gripper_open()

        # 复位
        await controller.move_init()

        # 获取状态
        state = await controller.get_state()
    """

    def __init__(self, config: RoArmConfig = None):
        self._config = config or RoArmConfig()
        self._serial_conn = None
        self._http_base_url = ""
        self._connected = False

        if self._config.mode == "http":
            self._http_base_url = f"http://{self._config.http_ip}:{self._config.http_port}"

    async def connect(self) -> bool:
        """建立连接"""
        try:
            if self._config.mode == "serial":
                import serial
                self._serial_conn = serial.Serial(
                    port=self._config.serial_port,
                    baudrate=self._config.serial_baudrate,
                    timeout=self._config.timeout,
                    dsrdtr=None
                )
                self._serial_conn.setRTS(False)
                self._serial_conn.setDTR(False)

            # 测试连接
            state = await self.get_state()
            if state:
                self._connected = True
                logger.info(f"RoArm-M3 连接成功 (mode={self._config.mode})")
                return True
            else:
                logger.warning("RoArm-M3 连接测试无响应，但继续尝试使用")
                self._connected = True
                return True

        except Exception as e:
            logger.error(f"RoArm-M3 连接失败: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """断开连接"""
        if self._serial_conn:
            self._serial_conn.close()
            self._serial_conn = None
        self._connected = False
        logger.info("RoArm-M3 已断开")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ============ 底层通信 ============

    async def _send_command(self, cmd_json: Dict[str, Any]) -> Optional[Dict]:
        """发送JSON命令并返回响应"""
        cmd_str = json.dumps(cmd_json, separators=(',', ':'))
        logger.debug(f"发送指令: {cmd_str}")

        try:
            if self._config.mode == "http":
                return await self._send_http(cmd_str)
            elif self._config.mode == "serial":
                return await self._send_serial(cmd_str)
            else:
                logger.error(f"未知通信模式: {self._config.mode}")
                return None
        except Exception as e:
            logger.error(f"指令发送失败 [{cmd_json.get('T')}]: {e}")
            return None

    async def _send_http(self, cmd_str: str) -> Optional[Dict]:
        """HTTP方式发送指令"""
        import urllib.parse
        url = f"{self._http_base_url}/js?json={urllib.parse.quote(cmd_str)}"

        loop = asyncio.get_event_loop()

        def _do_get():
            try:
                import requests
                resp = requests.get(url, timeout=self._config.timeout)
                text = resp.text.strip()
            except ImportError:
                import urllib.request
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                    text = resp.read().decode('utf-8').strip()
            if text:
                return json.loads(text)
            return {"status": "ok"}

        result = await loop.run_in_executor(None, _do_get)
        return result

    async def _send_serial(self, cmd_str: str) -> Optional[Dict]:
        """串口方式发送指令"""
        loop = asyncio.get_event_loop()

        def _do_serial():
            self._serial_conn.write((cmd_str + '\n').encode('utf-8'))
            self._serial_conn.flush()

            line = self._serial_conn.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return {"raw_response": line}
            return {"status": "ok"}

        result = await loop.run_in_executor(None, _do_serial)
        return result

    # ============ 运动控制 - 高级接口 ============

    async def move_init(self) -> bool:
        """
        移动到初始位置（复位）

        T=100 CMD_MOVE_INIT，会阻塞直到完成
        """
        result = await self._send_command({"T": 100})
        if result is not None:
            logger.info("机械臂复位完成")
            return True
        return False

    async def move_to_xyz(
        self,
        x: float, y: float, z: float,
        t: float = 0,
        r: float = 0,
        g: float = 3.14,
        spd: float = 0.25,
        blocking: bool = True
    ) -> bool:
        """
        移动末端到XYZ坐标位置（逆运动学）

        Args:
            x, y, z: 目标坐标 (mm)，基于右手坐标系
                      X: 机械臂正前方为正
                      Y: 机械臂左侧为正
                      Z: 垂直向上为正
            t: 末端俯仰角 (rad)
            r: 手腕Roll角 (rad)
            g: 夹爪角度 (rad), 3.14=闭合, 1.08=张开
            spd: 移动速度 (0-1+)
            blocking: 是否阻塞等待完成

        Returns:
            是否成功
        """
        cmd_type = 104 if blocking else 1041
        cmd = {"T": cmd_type, "x": x, "y": y, "z": z, "t": t, "r": r, "g": g}
        if blocking:
            cmd["spd"] = spd

        result = await self._send_command(cmd)

        if result is not None:
            logger.info(f"移动到 ({x}, {y}, {z}) 完成")
            return True
        return False

    async def move_to_xyz_nonblock(self, x: float, y: float, z: float,
                                     g: float = 3.14) -> bool:
        """非阻塞移动到坐标（适合连续路径）"""
        return await self.move_to_xyz(x, y, z, g=g, blocking=False)

    async def move_axis(self, axis: int, pos: float, spd: float = 0.25) -> bool:
        """
        控制单个轴向移动 (T=103)

        Args:
            axis: 轴编号 1=X, 2=Y, 3=Z, 4=T(pitch), 5=R(roll), 6=G(gripper)
            pos: 目标位置 (mm或rad)
            spd: 速度系数
        """
        result = await self._send_command({
            "T": 103,
            "axis": axis,
            "pos": pos,
            "spd": spd
        })
        if result is not None:
            logger.info(f"轴{axis}移动到{pos}完成")
            return True
        return False

    # ============ 关节控制 ============

    async def set_joint(self, joint: Union[int, Joint], rad: float,
                        spd: int = 0, acc: int = 10) -> bool:
        """
        控制单关节转动（弧度制 T=101）

        Args:
            joint: 关节编号(1-6) 或 Joint 枚举
            rad: 目标角度 (弧度)
            spd: 速度 (steps/s), 0=最大
            acc: 加速度 (100 steps/s^2单位), 越小越平滑
        """
        j = joint.value if isinstance(joint, Joint) else joint
        result = await self._send_command({
            "T": 101,
            "joint": j,
            "rad": rad,
            "spd": spd,
            "acc": acc
        })
        if result is not None:
            logger.info(f"关节{j}转到{rad:.3f}rad完成")
            return True
        return False

    async def set_all_joints(self, base: float = 0, shoulder: float = 0,
                              elbow: float = 1.57, wrist: float = 0,
                              roll: float = 0, hand: float = 3.14,
                              spd: int = 0, acc: int = 10) -> bool:
        """
        同时控制所有关节（弧度制 T=102）

        Args:
            base: 底座 (rad)
            shoulder: 肩 (rad)
            elbow: 肘 (rad)
            wrist: 手腕1 (rad)
            roll: 手腕2/Roll (rad)
            hand: 夹爪 (rad), 3.14=闭合, 1.08=张开
            spd: 速度, 0=最大
            acc: 加速度
        """
        result = await self._send_command({
            "T": 102,
            "base": base,
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "roll": roll,
            "hand": hand,
            "spd": spd,
            "acc": acc
        })
        if result is not None:
            logger.info("全关节设置完成")
            return True
        return False

    async def set_joint_angle(self, joint: Union[int, Joint], angle: float,
                               spd: int = 10, acc: int = 10) -> bool:
        """
        控制单关节转动（角度制 T=121）

        Args:
            joint: 关节编号(1-6) 或 Joint 枚举
            angle: 目标角度 (度)
            spd: 速度 (deg/s), 0=最大
            acc: 加速度 (deg/s^2)
        """
        j = joint.value if isinstance(joint, Joint) else joint
        result = await self._send_command({
            "T": 121,
            "joint": j,
            "angle": angle,
            "spd": spd,
            "acc": acc
        })
        if result is not None:
            logger.info(f"关节{j}转到{angle}度完成")
            return True
        return False

    async def set_all_joints_angle(self, b: int = 0, s: int = 0, e: int = 90,
                                    t: int = 0, r: int = 0, h: int = 180,
                                    spd: int = 10, acc: int = 10) -> bool:
        """同时控制所有关节（角度制 T=122）"""
        result = await self._send_command({
            "T": 122,
            "b": b, "s": s, "e": e,
            "t": t, "r": r, "h": h,
            "spd": spd, "acc": acc
        })
        if result is not None:
            logger.info("全关节(角度)设置完成")
            return True
        return False

    # ============ 夹爪控制 ============

    async def gripper_close(self) -> bool:
        """闭合夹爪 (默认闭合位置 3.14rad)"""
        result = await self._send_command({
            "T": 106,
            "cmd": 3.141593,
            "spd": 0,
            "acc": 0
        })
        if result is not None:
            logger.info("夹爪闭合")
            return True
        return False

    async def gripper_open(self) -> bool:
        """打开夹爪 (1.57rad 张开)"""
        result = await self._send_command({
            "T": 106,
            "cmd": 1.57,
            "spd": 0,
            "acc": 0
        })
        if result is not None:
            logger.info("夹爪张开")
            return True
        return False

    async def gripper_set(self, rad: float) -> bool:
        """精确设置夹爪开合度 (rad, 3.14=闭合, 1.08=最大张开)"""
        result = await self._send_command({
            "T": 106,
            "cmd": rad,
            "spd": 0,
            "acc": 0
        })
        if result is not None:
            logger.info(f"夹爪设置为{rad:.2f}rad")
            return True
        return False

    # ============ 系统控制 ============

    async def set_torque(self, on: bool = True) -> bool:
        """
        设置扭矩锁开关 (T=210)

        Args:
            on: True=开启(不能手动转), False=关闭(可手动转)
        """
        result = await self._send_command({"T": 210, "cmd": 1 if on else 0})
        if result is not None:
            logger.info(f"扭矩锁{'开启' if on else '关闭'}")
            return True
        return False

    async def led_on(self) -> bool:
        """开启LED灯"""
        result = await self._send_command({"T": 111, "cmd": 255})
        return result is not None

    async def led_off(self) -> bool:
        """关闭LED灯"""
        result = await self._send_command({"T": 111, "cmd": 0})
        return result is not None

    # ============ 连续运动 ============

    async def start_continuous_move(self, mode: int = 0, axis: int = 0,
                                     direction: int = 1, spd: int = 5) -> bool:
        """
        启动连续运动 (T=123)

        Args:
            mode: 0=角度模式, 1=坐标模式
            axis:
               角度模式: 1=底座, 2=肩, 3=肘, 4=手腕1, 5=手腕2, 6=夹爪
               坐标模式: 1=X, 2=Y, 3=Z, 4=T, 5=R, 6=G
            direction: 0=停止, 1=正向增加, 2=反向减小
            spd: 速度系数 (建议0-20)
        """
        result = await self._send_command({
            "T": 123,
            "m": mode,
            "axis": axis,
            "cmd": direction,
            "spd": spd
        })
        return result is not None

    async def stop_continuous_move(self, mode: int = 0, axis: int = 0) -> bool:
        """停止连续运动"""
        return await self.start_continuous_move(mode=mode, axis=axis, direction=0, spd=0)

    # ============ 状态查询 ============

    async def get_state(self) -> Optional[ArmState]:
        """
        获取机械臂完整状态 (T=105)

        包含：末端坐标、各关节角度、负载、电压等
        """
        raw = await self._send_command({"T": 105})
        if raw and "x" in raw:
            joints = [
                JointState(Joint.BASE, raw["b"], 0, raw.get("tB", 0), raw.get("torswitchB", 1) == 1),
                JointState(Joint.SHOULDER, raw["s"], 0, raw.get("tS", 0), raw.get("torswitchS", 1) == 1),
                JointState(Joint.ELBOW, raw["e"], 0, raw.get("tE", 0), raw.get("torswitchE", 1) == 1),
                JointState(Joint.WRIST, raw["t"], 0, raw.get("tT", 0), raw.get("torswitchT", 1) == 1),
                JointState(Joint.ROLL, raw["r"], 0, raw.get("tR", 0), raw.get("torswitchR", 1) == 1),
                JointState(Joint.GRIPPER, raw["g"], 0, raw.get("tG", 0), raw.get("torswitchG", 1) == 1),
            ]
            return ArmState(
                x=raw.get("x", 0),
                y=raw.get("y", 0),
                z=raw.get("z", 0),
                pitch=raw.get("tit", 0),
                base_rad=raw.get("b", 0),
                shoulder_rad=raw.get("s", 0),
                elbow_rad=raw.get("e", 0),
                wrist_rad=raw.get("t", 0),
                roll_rad=raw.get("r", 0),
                gripper_rad=raw.get("g", 0),
                voltage=raw.get("v", 0) * 0.01,
                joints=joints,
                timestamp=""
            )
        return None

    async def ping(self) -> bool:
        """检测机械臂是否在线"""
        state = await self.get_state()
        return state is not None


# ============ 全局单例 ============

_instance: Optional[RoArmController] = None


def get_roarm_config() -> RoArmConfig:
    """从全局配置加载机械臂配置"""
    from core.config import get_config
    cfg = get_config()
    arm_cfg = getattr(cfg, 'roarm', None)

    config = RoArmConfig()
    if arm_cfg:
        config.mode = getattr(arm_cfg, 'mode', 'http')
        config.http_ip = getattr(arm_cfg, 'http_ip', '192.168.4.1')
        config.http_port = getattr(arm_cfg, 'http_port', 80)
        config.serial_port = getattr(arm_cfg, 'serial_port', '')
        config.serial_baudrate = getattr(arm_cfg, 'serial_baudrate', 115200)
        config.timeout = getattr(arm_cfg, 'timeout', 5.0)

    return config


def get_roarm_controller() -> RoArmController:
    """获取全局机械臂控制器实例"""
    global _instance
    if _instance is None:
        _instance = RoArmController(get_roarm_config())
    return _instance
