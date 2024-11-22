// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: TargetRelative.proto

#ifndef PROTOBUF_INCLUDED_TargetRelative_2eproto
#define PROTOBUF_INCLUDED_TargetRelative_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_TargetRelative_2eproto 

namespace protobuf_TargetRelative_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_TargetRelative_2eproto
namespace sensor_msgs {
namespace msgs {
class TargetRelative;
class TargetRelativeDefaultTypeInternal;
extern TargetRelativeDefaultTypeInternal _TargetRelative_default_instance_;
}  // namespace msgs
}  // namespace sensor_msgs
namespace google {
namespace protobuf {
template<> ::sensor_msgs::msgs::TargetRelative* Arena::CreateMaybeMessage<::sensor_msgs::msgs::TargetRelative>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace sensor_msgs {
namespace msgs {

// ===================================================================

class TargetRelative : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:sensor_msgs.msgs.TargetRelative) */ {
 public:
  TargetRelative();
  virtual ~TargetRelative();

  TargetRelative(const TargetRelative& from);

  inline TargetRelative& operator=(const TargetRelative& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TargetRelative(TargetRelative&& from) noexcept
    : TargetRelative() {
    *this = ::std::move(from);
  }

  inline TargetRelative& operator=(TargetRelative&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const TargetRelative& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const TargetRelative* internal_default_instance() {
    return reinterpret_cast<const TargetRelative*>(
               &_TargetRelative_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(TargetRelative* other);
  friend void swap(TargetRelative& a, TargetRelative& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TargetRelative* New() const final {
    return CreateMaybeMessage<TargetRelative>(NULL);
  }

  TargetRelative* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<TargetRelative>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const TargetRelative& from);
  void MergeFrom(const TargetRelative& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(TargetRelative* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required int64 time_usec = 1;
  bool has_time_usec() const;
  void clear_time_usec();
  static const int kTimeUsecFieldNumber = 1;
  ::google::protobuf::int64 time_usec() const;
  void set_time_usec(::google::protobuf::int64 value);

  // required float pos_x = 2;
  bool has_pos_x() const;
  void clear_pos_x();
  static const int kPosXFieldNumber = 2;
  float pos_x() const;
  void set_pos_x(float value);

  // required float pos_y = 3;
  bool has_pos_y() const;
  void clear_pos_y();
  static const int kPosYFieldNumber = 3;
  float pos_y() const;
  void set_pos_y(float value);

  // optional double attitude_q_w = 5;
  bool has_attitude_q_w() const;
  void clear_attitude_q_w();
  static const int kAttitudeQWFieldNumber = 5;
  double attitude_q_w() const;
  void set_attitude_q_w(double value);

  // optional double attitude_q_x = 6;
  bool has_attitude_q_x() const;
  void clear_attitude_q_x();
  static const int kAttitudeQXFieldNumber = 6;
  double attitude_q_x() const;
  void set_attitude_q_x(double value);

  // optional double attitude_q_y = 7;
  bool has_attitude_q_y() const;
  void clear_attitude_q_y();
  static const int kAttitudeQYFieldNumber = 7;
  double attitude_q_y() const;
  void set_attitude_q_y(double value);

  // optional double attitude_q_z = 8;
  bool has_attitude_q_z() const;
  void clear_attitude_q_z();
  static const int kAttitudeQZFieldNumber = 8;
  double attitude_q_z() const;
  void set_attitude_q_z(double value);

  // required float pos_z = 4;
  bool has_pos_z() const;
  void clear_pos_z();
  static const int kPosZFieldNumber = 4;
  float pos_z() const;
  void set_pos_z(float value);

  // optional float yaw_std = 13;
  bool has_yaw_std() const;
  void clear_yaw_std();
  static const int kYawStdFieldNumber = 13;
  float yaw_std() const;
  void set_yaw_std(float value);

  // optional double orientation_q_w = 9;
  bool has_orientation_q_w() const;
  void clear_orientation_q_w();
  static const int kOrientationQWFieldNumber = 9;
  double orientation_q_w() const;
  void set_orientation_q_w(double value);

  // optional double orientation_q_x = 10;
  bool has_orientation_q_x() const;
  void clear_orientation_q_x();
  static const int kOrientationQXFieldNumber = 10;
  double orientation_q_x() const;
  void set_orientation_q_x(double value);

  // optional double orientation_q_y = 11;
  bool has_orientation_q_y() const;
  void clear_orientation_q_y();
  static const int kOrientationQYFieldNumber = 11;
  double orientation_q_y() const;
  void set_orientation_q_y(double value);

  // optional double orientation_q_z = 12;
  bool has_orientation_q_z() const;
  void clear_orientation_q_z();
  static const int kOrientationQZFieldNumber = 12;
  double orientation_q_z() const;
  void set_orientation_q_z(double value);

  // optional float std_x = 14;
  bool has_std_x() const;
  void clear_std_x();
  static const int kStdXFieldNumber = 14;
  float std_x() const;
  void set_std_x(float value);

  // optional float std_y = 15;
  bool has_std_y() const;
  void clear_std_y();
  static const int kStdYFieldNumber = 15;
  float std_y() const;
  void set_std_y(float value);

  // optional float std_z = 16;
  bool has_std_z() const;
  void clear_std_z();
  static const int kStdZFieldNumber = 16;
  float std_z() const;
  void set_std_z(float value);

  // @@protoc_insertion_point(class_scope:sensor_msgs.msgs.TargetRelative)
 private:
  void set_has_time_usec();
  void clear_has_time_usec();
  void set_has_pos_x();
  void clear_has_pos_x();
  void set_has_pos_y();
  void clear_has_pos_y();
  void set_has_pos_z();
  void clear_has_pos_z();
  void set_has_attitude_q_w();
  void clear_has_attitude_q_w();
  void set_has_attitude_q_x();
  void clear_has_attitude_q_x();
  void set_has_attitude_q_y();
  void clear_has_attitude_q_y();
  void set_has_attitude_q_z();
  void clear_has_attitude_q_z();
  void set_has_orientation_q_w();
  void clear_has_orientation_q_w();
  void set_has_orientation_q_x();
  void clear_has_orientation_q_x();
  void set_has_orientation_q_y();
  void clear_has_orientation_q_y();
  void set_has_orientation_q_z();
  void clear_has_orientation_q_z();
  void set_has_yaw_std();
  void clear_has_yaw_std();
  void set_has_std_x();
  void clear_has_std_x();
  void set_has_std_y();
  void clear_has_std_y();
  void set_has_std_z();
  void clear_has_std_z();

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  ::google::protobuf::int64 time_usec_;
  float pos_x_;
  float pos_y_;
  double attitude_q_w_;
  double attitude_q_x_;
  double attitude_q_y_;
  double attitude_q_z_;
  float pos_z_;
  float yaw_std_;
  double orientation_q_w_;
  double orientation_q_x_;
  double orientation_q_y_;
  double orientation_q_z_;
  float std_x_;
  float std_y_;
  float std_z_;
  friend struct ::protobuf_TargetRelative_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TargetRelative

// required int64 time_usec = 1;
inline bool TargetRelative::has_time_usec() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void TargetRelative::set_has_time_usec() {
  _has_bits_[0] |= 0x00000001u;
}
inline void TargetRelative::clear_has_time_usec() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void TargetRelative::clear_time_usec() {
  time_usec_ = GOOGLE_LONGLONG(0);
  clear_has_time_usec();
}
inline ::google::protobuf::int64 TargetRelative::time_usec() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.time_usec)
  return time_usec_;
}
inline void TargetRelative::set_time_usec(::google::protobuf::int64 value) {
  set_has_time_usec();
  time_usec_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.time_usec)
}

// required float pos_x = 2;
inline bool TargetRelative::has_pos_x() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void TargetRelative::set_has_pos_x() {
  _has_bits_[0] |= 0x00000002u;
}
inline void TargetRelative::clear_has_pos_x() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void TargetRelative::clear_pos_x() {
  pos_x_ = 0;
  clear_has_pos_x();
}
inline float TargetRelative::pos_x() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.pos_x)
  return pos_x_;
}
inline void TargetRelative::set_pos_x(float value) {
  set_has_pos_x();
  pos_x_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.pos_x)
}

// required float pos_y = 3;
inline bool TargetRelative::has_pos_y() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void TargetRelative::set_has_pos_y() {
  _has_bits_[0] |= 0x00000004u;
}
inline void TargetRelative::clear_has_pos_y() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void TargetRelative::clear_pos_y() {
  pos_y_ = 0;
  clear_has_pos_y();
}
inline float TargetRelative::pos_y() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.pos_y)
  return pos_y_;
}
inline void TargetRelative::set_pos_y(float value) {
  set_has_pos_y();
  pos_y_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.pos_y)
}

// required float pos_z = 4;
inline bool TargetRelative::has_pos_z() const {
  return (_has_bits_[0] & 0x00000080u) != 0;
}
inline void TargetRelative::set_has_pos_z() {
  _has_bits_[0] |= 0x00000080u;
}
inline void TargetRelative::clear_has_pos_z() {
  _has_bits_[0] &= ~0x00000080u;
}
inline void TargetRelative::clear_pos_z() {
  pos_z_ = 0;
  clear_has_pos_z();
}
inline float TargetRelative::pos_z() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.pos_z)
  return pos_z_;
}
inline void TargetRelative::set_pos_z(float value) {
  set_has_pos_z();
  pos_z_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.pos_z)
}

// optional double attitude_q_w = 5;
inline bool TargetRelative::has_attitude_q_w() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void TargetRelative::set_has_attitude_q_w() {
  _has_bits_[0] |= 0x00000008u;
}
inline void TargetRelative::clear_has_attitude_q_w() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void TargetRelative::clear_attitude_q_w() {
  attitude_q_w_ = 0;
  clear_has_attitude_q_w();
}
inline double TargetRelative::attitude_q_w() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.attitude_q_w)
  return attitude_q_w_;
}
inline void TargetRelative::set_attitude_q_w(double value) {
  set_has_attitude_q_w();
  attitude_q_w_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.attitude_q_w)
}

// optional double attitude_q_x = 6;
inline bool TargetRelative::has_attitude_q_x() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void TargetRelative::set_has_attitude_q_x() {
  _has_bits_[0] |= 0x00000010u;
}
inline void TargetRelative::clear_has_attitude_q_x() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void TargetRelative::clear_attitude_q_x() {
  attitude_q_x_ = 0;
  clear_has_attitude_q_x();
}
inline double TargetRelative::attitude_q_x() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.attitude_q_x)
  return attitude_q_x_;
}
inline void TargetRelative::set_attitude_q_x(double value) {
  set_has_attitude_q_x();
  attitude_q_x_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.attitude_q_x)
}

// optional double attitude_q_y = 7;
inline bool TargetRelative::has_attitude_q_y() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void TargetRelative::set_has_attitude_q_y() {
  _has_bits_[0] |= 0x00000020u;
}
inline void TargetRelative::clear_has_attitude_q_y() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void TargetRelative::clear_attitude_q_y() {
  attitude_q_y_ = 0;
  clear_has_attitude_q_y();
}
inline double TargetRelative::attitude_q_y() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.attitude_q_y)
  return attitude_q_y_;
}
inline void TargetRelative::set_attitude_q_y(double value) {
  set_has_attitude_q_y();
  attitude_q_y_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.attitude_q_y)
}

// optional double attitude_q_z = 8;
inline bool TargetRelative::has_attitude_q_z() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void TargetRelative::set_has_attitude_q_z() {
  _has_bits_[0] |= 0x00000040u;
}
inline void TargetRelative::clear_has_attitude_q_z() {
  _has_bits_[0] &= ~0x00000040u;
}
inline void TargetRelative::clear_attitude_q_z() {
  attitude_q_z_ = 0;
  clear_has_attitude_q_z();
}
inline double TargetRelative::attitude_q_z() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.attitude_q_z)
  return attitude_q_z_;
}
inline void TargetRelative::set_attitude_q_z(double value) {
  set_has_attitude_q_z();
  attitude_q_z_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.attitude_q_z)
}

// optional double orientation_q_w = 9;
inline bool TargetRelative::has_orientation_q_w() const {
  return (_has_bits_[0] & 0x00000200u) != 0;
}
inline void TargetRelative::set_has_orientation_q_w() {
  _has_bits_[0] |= 0x00000200u;
}
inline void TargetRelative::clear_has_orientation_q_w() {
  _has_bits_[0] &= ~0x00000200u;
}
inline void TargetRelative::clear_orientation_q_w() {
  orientation_q_w_ = 0;
  clear_has_orientation_q_w();
}
inline double TargetRelative::orientation_q_w() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.orientation_q_w)
  return orientation_q_w_;
}
inline void TargetRelative::set_orientation_q_w(double value) {
  set_has_orientation_q_w();
  orientation_q_w_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.orientation_q_w)
}

// optional double orientation_q_x = 10;
inline bool TargetRelative::has_orientation_q_x() const {
  return (_has_bits_[0] & 0x00000400u) != 0;
}
inline void TargetRelative::set_has_orientation_q_x() {
  _has_bits_[0] |= 0x00000400u;
}
inline void TargetRelative::clear_has_orientation_q_x() {
  _has_bits_[0] &= ~0x00000400u;
}
inline void TargetRelative::clear_orientation_q_x() {
  orientation_q_x_ = 0;
  clear_has_orientation_q_x();
}
inline double TargetRelative::orientation_q_x() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.orientation_q_x)
  return orientation_q_x_;
}
inline void TargetRelative::set_orientation_q_x(double value) {
  set_has_orientation_q_x();
  orientation_q_x_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.orientation_q_x)
}

// optional double orientation_q_y = 11;
inline bool TargetRelative::has_orientation_q_y() const {
  return (_has_bits_[0] & 0x00000800u) != 0;
}
inline void TargetRelative::set_has_orientation_q_y() {
  _has_bits_[0] |= 0x00000800u;
}
inline void TargetRelative::clear_has_orientation_q_y() {
  _has_bits_[0] &= ~0x00000800u;
}
inline void TargetRelative::clear_orientation_q_y() {
  orientation_q_y_ = 0;
  clear_has_orientation_q_y();
}
inline double TargetRelative::orientation_q_y() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.orientation_q_y)
  return orientation_q_y_;
}
inline void TargetRelative::set_orientation_q_y(double value) {
  set_has_orientation_q_y();
  orientation_q_y_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.orientation_q_y)
}

// optional double orientation_q_z = 12;
inline bool TargetRelative::has_orientation_q_z() const {
  return (_has_bits_[0] & 0x00001000u) != 0;
}
inline void TargetRelative::set_has_orientation_q_z() {
  _has_bits_[0] |= 0x00001000u;
}
inline void TargetRelative::clear_has_orientation_q_z() {
  _has_bits_[0] &= ~0x00001000u;
}
inline void TargetRelative::clear_orientation_q_z() {
  orientation_q_z_ = 0;
  clear_has_orientation_q_z();
}
inline double TargetRelative::orientation_q_z() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.orientation_q_z)
  return orientation_q_z_;
}
inline void TargetRelative::set_orientation_q_z(double value) {
  set_has_orientation_q_z();
  orientation_q_z_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.orientation_q_z)
}

// optional float yaw_std = 13;
inline bool TargetRelative::has_yaw_std() const {
  return (_has_bits_[0] & 0x00000100u) != 0;
}
inline void TargetRelative::set_has_yaw_std() {
  _has_bits_[0] |= 0x00000100u;
}
inline void TargetRelative::clear_has_yaw_std() {
  _has_bits_[0] &= ~0x00000100u;
}
inline void TargetRelative::clear_yaw_std() {
  yaw_std_ = 0;
  clear_has_yaw_std();
}
inline float TargetRelative::yaw_std() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.yaw_std)
  return yaw_std_;
}
inline void TargetRelative::set_yaw_std(float value) {
  set_has_yaw_std();
  yaw_std_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.yaw_std)
}

// optional float std_x = 14;
inline bool TargetRelative::has_std_x() const {
  return (_has_bits_[0] & 0x00002000u) != 0;
}
inline void TargetRelative::set_has_std_x() {
  _has_bits_[0] |= 0x00002000u;
}
inline void TargetRelative::clear_has_std_x() {
  _has_bits_[0] &= ~0x00002000u;
}
inline void TargetRelative::clear_std_x() {
  std_x_ = 0;
  clear_has_std_x();
}
inline float TargetRelative::std_x() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.std_x)
  return std_x_;
}
inline void TargetRelative::set_std_x(float value) {
  set_has_std_x();
  std_x_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.std_x)
}

// optional float std_y = 15;
inline bool TargetRelative::has_std_y() const {
  return (_has_bits_[0] & 0x00004000u) != 0;
}
inline void TargetRelative::set_has_std_y() {
  _has_bits_[0] |= 0x00004000u;
}
inline void TargetRelative::clear_has_std_y() {
  _has_bits_[0] &= ~0x00004000u;
}
inline void TargetRelative::clear_std_y() {
  std_y_ = 0;
  clear_has_std_y();
}
inline float TargetRelative::std_y() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.std_y)
  return std_y_;
}
inline void TargetRelative::set_std_y(float value) {
  set_has_std_y();
  std_y_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.std_y)
}

// optional float std_z = 16;
inline bool TargetRelative::has_std_z() const {
  return (_has_bits_[0] & 0x00008000u) != 0;
}
inline void TargetRelative::set_has_std_z() {
  _has_bits_[0] |= 0x00008000u;
}
inline void TargetRelative::clear_has_std_z() {
  _has_bits_[0] &= ~0x00008000u;
}
inline void TargetRelative::clear_std_z() {
  std_z_ = 0;
  clear_has_std_z();
}
inline float TargetRelative::std_z() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.TargetRelative.std_z)
  return std_z_;
}
inline void TargetRelative::set_std_z(float value) {
  set_has_std_z();
  std_z_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.TargetRelative.std_z)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace msgs
}  // namespace sensor_msgs

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_TargetRelative_2eproto
