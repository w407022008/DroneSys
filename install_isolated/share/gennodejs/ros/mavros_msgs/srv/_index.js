
"use strict";

let FileRead = require('./FileRead.js')
let SetMavFrame = require('./SetMavFrame.js')
let WaypointPush = require('./WaypointPush.js')
let MessageInterval = require('./MessageInterval.js')
let CommandHome = require('./CommandHome.js')
let WaypointSetCurrent = require('./WaypointSetCurrent.js')
let FileList = require('./FileList.js')
let ParamSet = require('./ParamSet.js')
let ParamGet = require('./ParamGet.js')
let FileWrite = require('./FileWrite.js')
let FileRemove = require('./FileRemove.js')
let CommandVtolTransition = require('./CommandVtolTransition.js')
let FileOpen = require('./FileOpen.js')
let CommandLong = require('./CommandLong.js')
let WaypointPull = require('./WaypointPull.js')
let FileRemoveDir = require('./FileRemoveDir.js')
let LogRequestList = require('./LogRequestList.js')
let FileChecksum = require('./FileChecksum.js')
let CommandTriggerInterval = require('./CommandTriggerInterval.js')
let FileMakeDir = require('./FileMakeDir.js')
let CommandAck = require('./CommandAck.js')
let StreamRate = require('./StreamRate.js')
let ParamPull = require('./ParamPull.js')
let LogRequestEnd = require('./LogRequestEnd.js')
let FileTruncate = require('./FileTruncate.js')
let VehicleInfoGet = require('./VehicleInfoGet.js')
let CommandBool = require('./CommandBool.js')
let CommandInt = require('./CommandInt.js')
let FileClose = require('./FileClose.js')
let FileRename = require('./FileRename.js')
let CommandTriggerControl = require('./CommandTriggerControl.js')
let WaypointClear = require('./WaypointClear.js')
let CommandTOL = require('./CommandTOL.js')
let SetMode = require('./SetMode.js')
let ParamPush = require('./ParamPush.js')
let LogRequestData = require('./LogRequestData.js')
let MountConfigure = require('./MountConfigure.js')

module.exports = {
  FileRead: FileRead,
  SetMavFrame: SetMavFrame,
  WaypointPush: WaypointPush,
  MessageInterval: MessageInterval,
  CommandHome: CommandHome,
  WaypointSetCurrent: WaypointSetCurrent,
  FileList: FileList,
  ParamSet: ParamSet,
  ParamGet: ParamGet,
  FileWrite: FileWrite,
  FileRemove: FileRemove,
  CommandVtolTransition: CommandVtolTransition,
  FileOpen: FileOpen,
  CommandLong: CommandLong,
  WaypointPull: WaypointPull,
  FileRemoveDir: FileRemoveDir,
  LogRequestList: LogRequestList,
  FileChecksum: FileChecksum,
  CommandTriggerInterval: CommandTriggerInterval,
  FileMakeDir: FileMakeDir,
  CommandAck: CommandAck,
  StreamRate: StreamRate,
  ParamPull: ParamPull,
  LogRequestEnd: LogRequestEnd,
  FileTruncate: FileTruncate,
  VehicleInfoGet: VehicleInfoGet,
  CommandBool: CommandBool,
  CommandInt: CommandInt,
  FileClose: FileClose,
  FileRename: FileRename,
  CommandTriggerControl: CommandTriggerControl,
  WaypointClear: WaypointClear,
  CommandTOL: CommandTOL,
  SetMode: SetMode,
  ParamPush: ParamPush,
  LogRequestData: LogRequestData,
  MountConfigure: MountConfigure,
};
