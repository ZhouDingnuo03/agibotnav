#!/bin/bash

# ===================== 第一步：定义全局变量（存储PID） =====================
declare -a PIDS=()
# 校准配置路径
ORIGINAL_YAML="/home/orin-001/sda/agibotnav/src/FAST_LIO/config/mid360.yaml"
CALIBRATED_YAML="/home/orin-001/sda/agibotnav/src/FAST_LIO/config/mid360_calibrated.yaml"
CALIBRATE_SCRIPT="/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/calibrate_imu_extrinsic.py"

# ===================== 第二步：进程清理函数（精准杀死PID） =====================
cleanup_all_processes() {
    echo -e "\n[INFO] 开始清理所有相关进程..."
    
    # 1. 强制杀死通过PID跟踪的进程
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "[INFO] 强制杀死跟踪的进程PID：${PIDS[*]}"
        kill -9 "${PIDS[@]}" 2>/dev/null
        PIDS=()
    fi
    
    # 杀死校准脚本进程
    pkill -9 -f "calibrate_imu_extrinsic.py" 2>/dev/null
    
    # 2. 兜底杀死静态TF发布器
    killall -9 static_transform_publisher 2>/dev/null
    
    # 3. 杀死ROS2相关进程
    ros2 daemon stop 2>/dev/null
    pkill -9 -f "ros2 " 2>/dev/null
    pkill -9 -f "python3.*ros" 2>/dev/null
    pkill -9 -f "fastlio_mapping" 2>/dev/null
    pkill -9 -f "fast_lio" 2>/dev/null
    pkill -9 -f "livox_ros_driver2" 2>/dev/null
    pkill -9 -f "ros2 launch" 2>/dev/null
    
    # 4. 精准杀死所有自定义Python脚本
    pkill -9 -f "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/path_planner.py" 2>/dev/null
    pkill -9 -f "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/grid_map_generator.py" 2>/dev/null
    pkill -9 -f "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/pcd.py" 2>/dev/null
    pkill -9 -f "/home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/search.py" 2>/dev/null
    
    # 5. 清理ROS2缓存
    rm -rf /tmp/ros2* /tmp/RMW* 2>/dev/null
    
    echo "[INFO] 所有进程清理完成！"
}

# ===================== 第三步：捕获退出信号 =====================
trap cleanup_all_processes SIGINT EXIT SIGTERM

# ===================== 第四步：校准结果检查函数 =====================
check_calibration_file() {
    if [ ! -f "${CALIBRATED_YAML}" ]; then
        echo "[WARNING] 校准文件不存在，复制原始配置"
        cp "${ORIGINAL_YAML}" "${CALIBRATED_YAML}"
    fi
}

# ===================== 第五步：启动逻辑（捕获每个进程PID） =====================
echo "[INFO] 开始启动所有节点..."

# 启动静态TF变换（捕获PID）
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link &
PIDS+=($!)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 body map &
PIDS+=($!)
sleep 1

# 先杀死旧进程
pkill -f fast_lio 2>/dev/null
pkill -f livox_ros_driver2 2>/dev/null

# 启动Livox MID360驱动（捕获PID）
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
PIDS+=($!)
echo "[INFO] 开始IMU外参水平校准..."
python3 "${CALIBRATE_SCRIPT}" --target "${ORIGINAL_YAML}"
# 检查校准文件（容错）
check_calibration_file
echo "[INFO] IMU校准脚本执行完成"
sleep 3
ros2 launch fast_lio mapping.launch.py &
PIDS+=($!)


# 启动自定义Python脚本（逐个捕获PID）
python3 /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/path_planner.py &
PIDS+=($!)
python3 /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/grid_map_generator.py &
PIDS+=($!)
python3 /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/pcd.py &
PIDS+=($!)
python3 /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/video.py &
PIDS+=($!)
sleep 10
python3 /home/orin-001/sda/agibotnav/src/FAST_LIO/scripts/search.py &
PIDS+=($!)

echo "[INFO] 所有节点启动完成！跟踪的进程PID：${PIDS[*]}"
echo "[INFO] 按 Ctrl+C 退出并强制清理所有进程"

# ===================== 第六步：保持脚本运行 =====================
wait

