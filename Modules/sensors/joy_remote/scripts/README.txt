JY901传感器信号接收步骤 

python文件：

linux中无法采用windows所用的com端口接收信号，但可用以下类似方法来完成，所用python库为Pyserial，
代码为官网提供，并稍作修改。

1，须确保Linux与蓝牙相连，做测试时所用的系统为linux虚拟机，因此需要一个外接dongle来实现linux的蓝牙接收。
打开蓝牙，右上角亮起蓝牙图标，连接传感器，anemometer_test显示状态为disconnect为正常状态，后进行下一步。


2，创建虚拟端口，
	ls /dev/rf* 查看现有端口  
	sudo rfcomm bind rfcomm0 00:14:03:06:0B:F2 为蓝牙创建一个名为rfcomm0的虚拟端口
	ls /dev/rf* 再次使用该命令查看现有端口，rfcomm0被添加即为成功。
	在安装cutecom后
	sudo cutecom 打开cutecom检验是否可以从rfcomm0端口接收信号。
	sudo chmod a+rw /dev/rfcomm0 或 sudo chmod 666 /dev/rfcomm0 为端口添加可读权限
	roscore
	python3 + 文件名 运行文件
	所创建的ros节点为wit/imu 
	rostopic echo 查看节点所接收的数据
	
	 
   

