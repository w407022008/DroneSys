
"use strict";

let ESCStatus = require('./ESCStatus.js');
let RCIn = require('./RCIn.js');
let VFR_HUD = require('./VFR_HUD.js');
let State = require('./State.js');
let DebugValue = require('./DebugValue.js');
let PlayTuneV2 = require('./PlayTuneV2.js');
let HilGPS = require('./HilGPS.js');
let StatusText = require('./StatusText.js');
let TerrainReport = require('./TerrainReport.js');
let ADSBVehicle = require('./ADSBVehicle.js');
let ManualControl = require('./ManualControl.js');
let HilControls = require('./HilControls.js');
let ESCInfoItem = require('./ESCInfoItem.js');
let NavControllerOutput = require('./NavControllerOutput.js');
let RTKBaseline = require('./RTKBaseline.js');
let Thrust = require('./Thrust.js');
let LandingTarget = require('./LandingTarget.js');
let Trajectory = require('./Trajectory.js');
let RTCM = require('./RTCM.js');
let ExtendedState = require('./ExtendedState.js');
let GPSINPUT = require('./GPSINPUT.js');
let ParamValue = require('./ParamValue.js');
let PositionTarget = require('./PositionTarget.js');
let HilSensor = require('./HilSensor.js');
let CellularStatus = require('./CellularStatus.js');
let CompanionProcessStatus = require('./CompanionProcessStatus.js');
let ESCTelemetry = require('./ESCTelemetry.js');
let OverrideRCIn = require('./OverrideRCIn.js');
let GPSRTK = require('./GPSRTK.js');
let RadioStatus = require('./RadioStatus.js');
let SysStatus = require('./SysStatus.js');
let ESCStatusItem = require('./ESCStatusItem.js');
let OpticalFlowRad = require('./OpticalFlowRad.js');
let OnboardComputerStatus = require('./OnboardComputerStatus.js');
let CommandCode = require('./CommandCode.js');
let Waypoint = require('./Waypoint.js');
let HilStateQuaternion = require('./HilStateQuaternion.js');
let ESCInfo = require('./ESCInfo.js');
let WheelOdomStamped = require('./WheelOdomStamped.js');
let WaypointReached = require('./WaypointReached.js');
let LogData = require('./LogData.js');
let EstimatorStatus = require('./EstimatorStatus.js');
let HomePosition = require('./HomePosition.js');
let TimesyncStatus = require('./TimesyncStatus.js');
let VehicleInfo = require('./VehicleInfo.js');
let CameraImageCaptured = require('./CameraImageCaptured.js');
let BatteryStatus = require('./BatteryStatus.js');
let RCOut = require('./RCOut.js');
let Param = require('./Param.js');
let GlobalPositionTarget = require('./GlobalPositionTarget.js');
let Vibration = require('./Vibration.js');
let HilActuatorControls = require('./HilActuatorControls.js');
let WaypointList = require('./WaypointList.js');
let MagnetometerReporter = require('./MagnetometerReporter.js');
let Altitude = require('./Altitude.js');
let CamIMUStamp = require('./CamIMUStamp.js');
let FileEntry = require('./FileEntry.js');
let Tunnel = require('./Tunnel.js');
let LogEntry = require('./LogEntry.js');
let MountControl = require('./MountControl.js');
let Mavlink = require('./Mavlink.js');
let AttitudeTarget = require('./AttitudeTarget.js');
let ESCTelemetryItem = require('./ESCTelemetryItem.js');
let ActuatorControl = require('./ActuatorControl.js');
let GPSRAW = require('./GPSRAW.js');

module.exports = {
  ESCStatus: ESCStatus,
  RCIn: RCIn,
  VFR_HUD: VFR_HUD,
  State: State,
  DebugValue: DebugValue,
  PlayTuneV2: PlayTuneV2,
  HilGPS: HilGPS,
  StatusText: StatusText,
  TerrainReport: TerrainReport,
  ADSBVehicle: ADSBVehicle,
  ManualControl: ManualControl,
  HilControls: HilControls,
  ESCInfoItem: ESCInfoItem,
  NavControllerOutput: NavControllerOutput,
  RTKBaseline: RTKBaseline,
  Thrust: Thrust,
  LandingTarget: LandingTarget,
  Trajectory: Trajectory,
  RTCM: RTCM,
  ExtendedState: ExtendedState,
  GPSINPUT: GPSINPUT,
  ParamValue: ParamValue,
  PositionTarget: PositionTarget,
  HilSensor: HilSensor,
  CellularStatus: CellularStatus,
  CompanionProcessStatus: CompanionProcessStatus,
  ESCTelemetry: ESCTelemetry,
  OverrideRCIn: OverrideRCIn,
  GPSRTK: GPSRTK,
  RadioStatus: RadioStatus,
  SysStatus: SysStatus,
  ESCStatusItem: ESCStatusItem,
  OpticalFlowRad: OpticalFlowRad,
  OnboardComputerStatus: OnboardComputerStatus,
  CommandCode: CommandCode,
  Waypoint: Waypoint,
  HilStateQuaternion: HilStateQuaternion,
  ESCInfo: ESCInfo,
  WheelOdomStamped: WheelOdomStamped,
  WaypointReached: WaypointReached,
  LogData: LogData,
  EstimatorStatus: EstimatorStatus,
  HomePosition: HomePosition,
  TimesyncStatus: TimesyncStatus,
  VehicleInfo: VehicleInfo,
  CameraImageCaptured: CameraImageCaptured,
  BatteryStatus: BatteryStatus,
  RCOut: RCOut,
  Param: Param,
  GlobalPositionTarget: GlobalPositionTarget,
  Vibration: Vibration,
  HilActuatorControls: HilActuatorControls,
  WaypointList: WaypointList,
  MagnetometerReporter: MagnetometerReporter,
  Altitude: Altitude,
  CamIMUStamp: CamIMUStamp,
  FileEntry: FileEntry,
  Tunnel: Tunnel,
  LogEntry: LogEntry,
  MountControl: MountControl,
  Mavlink: Mavlink,
  AttitudeTarget: AttitudeTarget,
  ESCTelemetryItem: ESCTelemetryItem,
  ActuatorControl: ActuatorControl,
  GPSRAW: GPSRAW,
};
