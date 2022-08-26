#!/usr/bin/env python
import pyrealsense2 as rs

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
config = rs.config()
pipeline_profile = pipeline.start(config)
device = pipeline_profile.get_device()
depth_sensor = device.query_sensors()

print(depth_sensor[0].get_option(rs.option.emitter_on_off))
#depth_sensor[0].set_option(rs.option.emitter_enabled, 1)
depth_sensor[0].set_option(rs.option.emitter_on_off,1)
depth_sensor[0].get_option(rs.option.emitter_enabled)

try:
    while True:
        emitter = depth_sensor[0].get_option(rs.option.emitter_enabled)
        print("emitter = ", emitter)

finally:
    pipeline.stop()

