std_msgs/Header header

uint8 message_type
# enum message_type
uint8 AIRFLOW = 0  
uint8 FORCE   = 1  
uint8 IMU  = 2  

## Airflow Measurement
uint8 airflow_sensor_num        ## the number of airflow sensor
float32[16] current             ## airflow sensor current measurement [mA]
float32[16] voltage             ## airflow sensor voltage measurement [mV]
float32[16] power               ## airflow sensor power measurement [mW]
float32[8] pow_diff             ## airflow sensor power measurement difference [mW]

## Force Measurement
float32[4] diff_volt            ## Bridge voltage difference of force sensor [uV]

## IMU Measurement
geometry_msgs/Quaternion quaternion			## Quaternion rotation from XYZ body frame to ENU earth frame.
geometry_msgs/Vector3 eular_angle			## Eular angle rotation from XYZ body frame to ENU earth frame.
geometry_msgs/Vector3 acc                  ## in XYZ body frame[m/s^2]
geometry_msgs/Vector3 mag                  ## [m/s^2]
geometry_msgs/Vector3 gyro                 ## [m/s^2]
uint32 baro                      ## [pascal]
float32 temp                     ## [degree]
