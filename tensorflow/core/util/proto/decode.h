/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Inline functions for parsing the protocol buffers wire format.
//
// These functions have been optimized at the expense of safety.
// They are broken out into a separate file for readability but are
// not intended for use by clients other than the decode_proto op.
//
// The calling code in the decode_proto op does some fairly
// complicated things to ensure that this code is called
// safely. Changes to this code should be thoroughly fuzz tested.

#ifndef TENSORFLOW_CORE_UTIL_PROTO_DECODE_H_
#define TENSORFLOW_CORE_UTIL_PROTO_DECODE_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace internal {

using tensorflow::protobuf::internal::WireFormatLite;
using tensorflow::protobuf::io::CodedInputStream;
using tensorflow::protobuf::io::CodedOutputStream;
using tensorflow::protobuf::io::StringOutputStream;

// Converts an uint64 to an int64 without loss of information.
// Unsigned values greater than INT64_MAX are represented as
// negative numbers by wrapping (same as twos-complement bit equivalence).
inline int64_t WrapUnsignedAsSigned64(uint64 unsigned_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/util/proto/decode.h", "WrapUnsignedAsSigned64");

  // For a detailed explanation of why this works to wrap unsigned ints, see
  // http://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
  // Both if tests should be optimized out.
  if (unsigned_value <= INT64_MAX) {
    return static_cast<int64_t>(unsigned_value);
  }
  // The C++ spec allows an architecture where this test is required.
  if (unsigned_value >= INT64_MIN) {
    return static_cast<int64_t>(unsigned_value - INT64_MIN) + INT64_MIN;
  }
  return 0;  // This should never occur.
}

// Converts an uint32 to an int32 without loss of information.
// Unsigned values greater than INT_MAX are represented as
// negative numbers by wrapping (same as twos-complement bit equivalence).
inline int32 WrapUnsignedAsSigned32(uint32 unsigned_value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_1(mht_1_v, 234, "", "./tensorflow/core/util/proto/decode.h", "WrapUnsignedAsSigned32");

  // For a detailed explanation of why this works to wrap unsigned ints, see
  // http://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
  // Both if tests should be optimized out.
  if (unsigned_value <= INT_MAX) {
    return static_cast<int32>(unsigned_value);
  }
  // The C++ spec allows an architecture where this test is required.
  if (unsigned_value >= INT_MIN) {
    return static_cast<int32>(unsigned_value - INT_MIN) + INT_MIN;
  }
  return 0;  // This should never occur.
}

// Reads a single varint32 from a byte array.
// It is the caller's responsibility to ensure that there is enough
// space in the buffer.
// The ok value will be set to false if the buffer does not contain
// a valid varint.
inline const uint8* ReadVarint64FromArray(const uint8* buffer, bool* ok,
                                          uint64* value);

// Reads a single varint32 from a byte array.
// It is the caller's responsibility to ensure that there is enough
// space in the buffer.
// The ok value will be set to false if the buffer does not contain
// a valid varint.
// This is slightly less efficient than the private version in
// coded_stream.cc but we duplicate less code by calling
// the 64 bit version instead of copying the code.
inline const uint8* ReadVarint32FromArray(const uint8* buffer, bool* ok,
                                          uint32* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_2(mht_2_v, 268, "", "./tensorflow/core/util/proto/decode.h", "ReadVarint32FromArray");

  uint64 tmp = 0;
  const uint8* buf = ReadVarint64FromArray(buffer, ok, &tmp);
  *value = tmp & 0xffffffff;
  return buf;
}

// Reads a single proto field value from a byte array into an array.
// The array is part of a Tensor that was allocated by the caller
// with type TensorType, while DeclaredType is the proto field type.
template <class TensorType, enum WireFormatLite::FieldType DeclaredType>
const uint8* ReadFromArray(const uint8* buf, TensorType* value);

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_INT32>(
    const uint8* buf, int64_t* value) {
  uint32 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint32FromArray(buf, &unused_ok, &temp);
  *value = static_cast<int64_t>(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<int32, WireFormatLite::TYPE_INT32>(
    const uint8* buf, int32* value) {
  uint32 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint32FromArray(buf, &unused_ok, &temp);
  *value = static_cast<int32>(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_INT64>(
    const uint8* buf, int64_t* value) {
  uint64 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint64FromArray(buf, &unused_ok, &temp);
  *value = WrapUnsignedAsSigned64(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<uint64, WireFormatLite::TYPE_UINT32>(
    const uint8* buf, uint64* value) {
  uint32 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint32FromArray(buf, &unused_ok, &temp);
  *value = temp;
  return buf;
}

template <>
inline const uint8* ReadFromArray<uint32, WireFormatLite::TYPE_UINT32>(
    const uint8* buf, uint32* value) {
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  return ReadVarint32FromArray(buf, &unused_ok, value);
}

template <>
inline const uint8* ReadFromArray<uint64, WireFormatLite::TYPE_UINT64>(
    const uint8* buf, uint64* value) {
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  return ReadVarint64FromArray(buf, &unused_ok, value);
}

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_SINT32>(
    const uint8* buf, int64_t* value) {
  uint64 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint64FromArray(buf, &unused_ok, &temp);
  *value = WireFormatLite::ZigZagDecode32(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<int32, WireFormatLite::TYPE_SINT32>(
    const uint8* buf, int32* value) {
  uint32 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint32FromArray(buf, &unused_ok, &temp);
  *value = WireFormatLite::ZigZagDecode32(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_SINT64>(
    const uint8* buf, int64_t* value) {
  uint64 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint64FromArray(buf, &unused_ok, &temp);
  *value = WireFormatLite::ZigZagDecode64(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<uint64, WireFormatLite::TYPE_FIXED32>(
    const uint8* buf, uint64* value) {
  uint32 temp;
  buf = WireFormatLite::ReadPrimitiveFromArray<uint32,
                                               WireFormatLite::TYPE_FIXED32>(
      buf, &temp);
  *value = temp;
  return buf;
}

template <>
inline const uint8* ReadFromArray<uint32, WireFormatLite::TYPE_FIXED32>(
    const uint8* buf, uint32* value) {
  uint32 temp;
  buf = WireFormatLite::ReadPrimitiveFromArray<uint32,
                                               WireFormatLite::TYPE_FIXED32>(
      buf, &temp);
  *value = WrapUnsignedAsSigned32(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<uint64, WireFormatLite::TYPE_FIXED64>(
    const uint8* buf, uint64* value) {
  protobuf_uint64 temp;
  buf = WireFormatLite::ReadPrimitiveFromArray<protobuf_uint64,
                                               WireFormatLite::TYPE_FIXED64>(
      buf, &temp);
  *value = WrapUnsignedAsSigned64(temp);
  return buf;
}

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_SFIXED32>(
    const uint8* buf, int64_t* value) {
  int32_t temp;
  buf = WireFormatLite::ReadPrimitiveFromArray<int32,
                                               WireFormatLite::TYPE_SFIXED32>(
      buf, &temp);
  *value = temp;
  return buf;
}

template <>
inline const uint8* ReadFromArray<int32, WireFormatLite::TYPE_SFIXED32>(
    const uint8* buf, int32* value) {
  return WireFormatLite::ReadPrimitiveFromArray<int32,
                                                WireFormatLite::TYPE_SFIXED32>(
      buf, value);
}

template <>
inline const uint8* ReadFromArray<int64_t, WireFormatLite::TYPE_SFIXED64>(
    const uint8* buf, int64_t* value) {
  protobuf_int64 temp;
  buf = WireFormatLite::ReadPrimitiveFromArray<protobuf_int64,
                                               WireFormatLite::TYPE_SFIXED64>(
      buf, &temp);
  *value = temp;
  return buf;
}

template <>
inline const uint8* ReadFromArray<float, WireFormatLite::TYPE_FLOAT>(
    const uint8* buf, float* value) {
  return WireFormatLite::ReadPrimitiveFromArray<float,
                                                WireFormatLite::TYPE_FLOAT>(
      buf, value);
}

template <>
inline const uint8* ReadFromArray<double, WireFormatLite::TYPE_FLOAT>(
    const uint8* buf, double* value) {
  float temp;
  buf =
      WireFormatLite::ReadPrimitiveFromArray<float, WireFormatLite::TYPE_FLOAT>(
          buf, &temp);
  *value = temp;
  return buf;
}

template <>
inline const uint8* ReadFromArray<double, WireFormatLite::TYPE_DOUBLE>(
    const uint8* buf, double* value) {
  return WireFormatLite::ReadPrimitiveFromArray<double,
                                                WireFormatLite::TYPE_DOUBLE>(
      buf, value);
}

template <>
inline const uint8* ReadFromArray<bool, WireFormatLite::TYPE_BOOL>(
    const uint8* buf, bool* value) {
  uint64 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint64FromArray(buf, &unused_ok, &temp);
  *value = temp != 0;
  return buf;
}

template <>
inline const uint8* ReadFromArray<int, WireFormatLite::TYPE_ENUM>(
    const uint8* buf, int* value) {
  uint32 temp = 0;
  bool unused_ok;  // The Counting pass would have failed if this were corrupt.
  buf = ReadVarint32FromArray(buf, &unused_ok, &temp);
  *value = static_cast<int>(temp);
  return buf;
}

// Reads packed values from an array.
// Stride is set to 1 for repeated fields, and 0 for non-repeated fields
// (where any value overwrites previous values).
template <class TensorType, enum WireFormatLite::FieldType DeclaredType>
inline int ReadPackedPrimitives(const void* bufp, const size_t len,
                                const int index, const int stride,
                                void* datap) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_3(mht_3_v, 484, "", "./tensorflow/core/util/proto/decode.h", "ReadPackedPrimitives");

  const uint8* buf = reinterpret_cast<const uint8*>(bufp);
  const uint8* bound = buf + len;
  TensorType* data = reinterpret_cast<TensorType*>(datap) + index;
  int count;

  // This could overrun the bound by stride-1. This is defended
  // against in the caller, where it ensures that the input buffer
  // contains complete values.
  for (count = 0; buf < bound; count += stride) {
    buf = ReadFromArray<TensorType, DeclaredType>(buf, data + count);
  }
  return count;
}

// Reads a value of a primitive type field from a serialized proto.
// The value is parsed from the serialized format, then static_cast
// to the desired type for TensorFlow and stored.
template <class ValueType, class TensorType,
          enum WireFormatLite::FieldType DeclaredType>
inline Status ReadPrimitive(CodedInputStream* input, int index, void* data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_4(mht_4_v, 507, "", "./tensorflow/core/util/proto/decode.h", "ReadPrimitive");

  ValueType v;
  if (!WireFormatLite::ReadPrimitive<ValueType, DeclaredType>(input, &v)) {
    return errors::DataLoss("Failed reading primitive");
  }

  reinterpret_cast<TensorType*>(data)[index] = v;
  return Status::OK();
}

// Reads a string, submessage, or other variable-length field from a
// serialized proto.
// May read all or part of a repeated field.
inline Status ReadBytes(CodedInputStream* input, int index, void* datap) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_5(mht_5_v, 523, "", "./tensorflow/core/util/proto/decode.h", "ReadBytes");

  tstring* data = reinterpret_cast<tstring*>(datap) + index;

  uint32 length;
  if (!input->ReadVarint32(&length)) {
    return errors::DataLoss("Failed reading bytes");
  }

  data->resize_uninitialized(length);

  if (!input->ReadRaw(data->data(), length)) {
    return errors::DataLoss("Failed reading bytes");
  }
  return Status::OK();
}

// Reads a tag-delimited field (TYPE_GROUP) from a serialized proto,
// as a bytestring.
inline Status ReadGroupBytes(CodedInputStream* input, int field_number,
                             int index, void* datap) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_6(mht_6_v, 545, "", "./tensorflow/core/util/proto/decode.h", "ReadGroupBytes");

  // WireFormatLite::SkipField has an option to emit the
  // skipped bytes to an output stream. We could do better by implementing our
  // own scanner but this is simpler for now.
  // TODO(nix): there is a faster way to grab TYPE_GROUP bytes by relying
  // on input->IsFlat() == true and using input->GetDirectBufferPointer()
  // with input->CurrentPosition().
  tstring* data = reinterpret_cast<tstring*>(datap) + index;
  // TODO(dero): To mitigate the string to tstring copy, we can implement our
  // own scanner as described above.  We would first need to obtain the length
  // in an initial pass and resize/reserve the tstring. But, given that
  // TYPE_GROUP is deprecated and currently no tests in
  // tensorflow/python/kernel_tests/proto:decode_proto_op_test target a
  // TYPE_GROUP tag, we use std::string as a read buffer.
  string buf;
  StringOutputStream string_stream(&buf);
  {
    CodedOutputStream out(&string_stream);
    if (!WireFormatLite::SkipField(
            input,
            WireFormatLite::MakeTag(field_number,
                                    WireFormatLite::WIRETYPE_START_GROUP),
            &out)) {
      return errors::DataLoss("Failed reading group");
    }
  }
  *data = buf;
  return Status::OK();
}

// Reads a single field value from a CodedInputStream into a tensor.
inline Status ReadValue(CodedInputStream* input,
                        WireFormatLite::FieldType field_type, int field_number,
                        DataType dtype, int index, void* datap) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_7(mht_7_v, 581, "", "./tensorflow/core/util/proto/decode.h", "ReadValue");

  // Dispatch to the appropriately typed field reader based on the schema type.
  switch (field_type) {
    case WireFormatLite::TYPE_DOUBLE:
      return ReadPrimitive<double, double, WireFormatLite::TYPE_DOUBLE>(
          input, index, datap);
    case WireFormatLite::TYPE_FLOAT:
      switch (dtype) {
        case DataType::DT_DOUBLE:
          return ReadPrimitive<float, double, WireFormatLite::TYPE_FLOAT>(
              input, index, datap);
        case DataType::DT_FLOAT:
          return ReadPrimitive<float, float, WireFormatLite::TYPE_FLOAT>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_FLOAT for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_INT64:
      return ReadPrimitive<protobuf_int64, int64_t, WireFormatLite::TYPE_INT64>(
          input, index, datap);
    case WireFormatLite::TYPE_UINT64:
      return ReadPrimitive<protobuf_uint64, uint64,
                           WireFormatLite::TYPE_UINT64>(input, index, datap);
    case WireFormatLite::TYPE_INT32:
      switch (dtype) {
        case DataType::DT_INT64:
          return ReadPrimitive<int32, int64_t, WireFormatLite::TYPE_INT32>(
              input, index, datap);
        case DataType::DT_INT32:
          return ReadPrimitive<int32, int32, WireFormatLite::TYPE_INT32>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_INT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_FIXED64:
      return ReadPrimitive<protobuf_uint64, uint64,
                           WireFormatLite::TYPE_FIXED64>(input, index, datap);
    case WireFormatLite::TYPE_FIXED32:
      switch (dtype) {
        case DataType::DT_UINT64:
          return ReadPrimitive<uint32, uint64, WireFormatLite::TYPE_FIXED32>(
              input, index, datap);
        case DataType::DT_UINT32:
          return ReadPrimitive<uint32, uint32, WireFormatLite::TYPE_FIXED32>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_FIXED32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_BOOL:
      return ReadPrimitive<bool, bool, WireFormatLite::TYPE_BOOL>(input, index,
                                                                  datap);
    case WireFormatLite::TYPE_STRING:
      return ReadBytes(input, index, datap);
    case WireFormatLite::TYPE_GROUP:
      return ReadGroupBytes(input, field_number, index, datap);
    case WireFormatLite::TYPE_MESSAGE:
      return ReadBytes(input, index, datap);
    case WireFormatLite::TYPE_BYTES:
      return ReadBytes(input, index, datap);
    case WireFormatLite::TYPE_UINT32:
      switch (dtype) {
        case DataType::DT_UINT64:
          return ReadPrimitive<uint32, uint64, WireFormatLite::TYPE_UINT32>(
              input, index, datap);
        case DataType::DT_UINT32:
          return ReadPrimitive<uint32, uint32, WireFormatLite::TYPE_UINT32>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_UINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_ENUM:
      return ReadPrimitive<int32, int32, WireFormatLite::TYPE_ENUM>(
          input, index, datap);
    case WireFormatLite::TYPE_SFIXED32:
      switch (dtype) {
        case DataType::DT_INT64:
          return ReadPrimitive<int32, int64_t, WireFormatLite::TYPE_SFIXED32>(
              input, index, datap);
        case DataType::DT_INT32:
          return ReadPrimitive<int32, int32, WireFormatLite::TYPE_SFIXED32>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_SFIXED32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SFIXED64:
      return ReadPrimitive<protobuf_int64, int64_t,
                           WireFormatLite::TYPE_SFIXED64>(input, index, datap);
    case WireFormatLite::TYPE_SINT32:
      switch (dtype) {
        case DataType::DT_INT64:
          return ReadPrimitive<int32, int64_t, WireFormatLite::TYPE_SINT32>(
              input, index, datap);
        case DataType::DT_INT32:
          return ReadPrimitive<int32, int32, WireFormatLite::TYPE_SINT32>(
              input, index, datap);
        default:
          return errors::DataLoss("Failed reading TYPE_SINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SINT64:
      return ReadPrimitive<protobuf_int64, int64_t,
                           WireFormatLite::TYPE_SINT64>(input, index, datap);
      // default: intentionally omitted in order to enable static checking.
  }
  // Unreachable.
  return errors::DataLoss("Failed reading unknown wire type");
}

// Reads and stores a length-delimited list of values.
inline Status ReadPackedFromArray(const void* buf, size_t buf_size,
                                  const WireFormatLite::FieldType field_type,
                                  const int field_number, const DataType dtype,
                                  const int stride, int* index, void* data) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_8(mht_8_v, 701, "", "./tensorflow/core/util/proto/decode.h", "ReadPackedFromArray");

  // Dispatch to the appropriately typed field reader based on the schema type.
  switch (field_type) {
    case WireFormatLite::TYPE_DOUBLE:
      *index += ReadPackedPrimitives<double, WireFormatLite::TYPE_DOUBLE>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_FLOAT:
      switch (dtype) {
        case DataType::DT_DOUBLE:
          *index += ReadPackedPrimitives<double, WireFormatLite::TYPE_FLOAT>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_FLOAT:
          *index += ReadPackedPrimitives<float, WireFormatLite::TYPE_FLOAT>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_FLOAT for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_INT64:
      *index += ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_INT64>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_UINT64:
      *index += ReadPackedPrimitives<uint64, WireFormatLite::TYPE_UINT64>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_INT32:
      switch (dtype) {
        case DataType::DT_INT64:
          *index += ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_INT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_INT32:
          *index += ReadPackedPrimitives<int32, WireFormatLite::TYPE_INT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_INT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_FIXED64:
      *index += ReadPackedPrimitives<uint64, WireFormatLite::TYPE_FIXED64>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_FIXED32:
      switch (dtype) {
        case DataType::DT_UINT64:
          *index += ReadPackedPrimitives<uint64, WireFormatLite::TYPE_FIXED32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_UINT32:
          *index += ReadPackedPrimitives<uint32, WireFormatLite::TYPE_FIXED32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_FIXED32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_BOOL:
      *index += ReadPackedPrimitives<bool, WireFormatLite::TYPE_BOOL>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_STRING:
    case WireFormatLite::TYPE_GROUP:
    case WireFormatLite::TYPE_MESSAGE:
    case WireFormatLite::TYPE_BYTES:
      return errors::DataLoss("Non-primitive type encountered as packed");
    case WireFormatLite::TYPE_UINT32:
      switch (dtype) {
        case DataType::DT_UINT64:
          *index += ReadPackedPrimitives<uint64, WireFormatLite::TYPE_UINT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_UINT32:
          *index += ReadPackedPrimitives<uint32, WireFormatLite::TYPE_UINT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_UINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_ENUM:
      *index += ReadPackedPrimitives<int32, WireFormatLite::TYPE_ENUM>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
    case WireFormatLite::TYPE_SFIXED32:
      switch (dtype) {
        case DataType::DT_INT64:
          *index +=
              ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_SFIXED32>(
                  buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_INT32:
          *index += ReadPackedPrimitives<int32, WireFormatLite::TYPE_SFIXED32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_INT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SFIXED64:
      *index += ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_SFIXED64>(
          buf, buf_size, *index, stride, data);
      return Status::OK();

    case WireFormatLite::TYPE_SINT32:
      switch (dtype) {
        case DataType::DT_INT64:
          *index += ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_SINT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        case DataType::DT_INT32:
          *index += ReadPackedPrimitives<int32, WireFormatLite::TYPE_SINT32>(
              buf, buf_size, *index, stride, data);
          return Status::OK();
        default:
          return errors::DataLoss("Failed reading TYPE_SINT32 for ",
                                  DataTypeString(dtype));
      }
    case WireFormatLite::TYPE_SINT64:
      *index += ReadPackedPrimitives<int64_t, WireFormatLite::TYPE_SINT64>(
          buf, buf_size, *index, stride, data);
      return Status::OK();
      // default: intentionally omitted in order to enable static checking.
  }
  // Unreachable.
  return errors::DataLoss("Failed reading unknown wire type");
}

// Reads a varint from the given buffer, write it to *value, and return the
// new buffer pointer.
// This was copied from coded_stream.cc where it is private.
// Important: This routine may read as much as kMaxVarintBytes from
// the buffer. It is the caller's responsibility to make sure that there is
// enough space in the buffer.
inline const uint8* ReadVarint64FromArray(const uint8* buffer, bool* ok,
                                          uint64* value) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSprotoPSdecodeDTh mht_9(mht_9_v, 843, "", "./tensorflow/core/util/proto/decode.h", "ReadVarint64FromArray");

  const uint8* ptr = buffer;
  uint32 b;

  // Splitting into 32-bit pieces gives better performance on 32-bit
  // processors.
  uint32 part0 = 0, part1 = 0, part2 = 0;

  b = *(ptr++);
  part0 = b;
  if (!(b & 0x80)) goto done;
  part0 -= 0x80;
  b = *(ptr++);
  part0 += b << 7;
  if (!(b & 0x80)) goto done;
  part0 -= 0x80 << 7;
  b = *(ptr++);
  part0 += b << 14;
  if (!(b & 0x80)) goto done;
  part0 -= 0x80 << 14;
  b = *(ptr++);
  part0 += b << 21;
  if (!(b & 0x80)) goto done;
  part0 -= 0x80 << 21;
  b = *(ptr++);
  part1 = b;
  if (!(b & 0x80)) goto done;
  part1 -= 0x80;
  b = *(ptr++);
  part1 += b << 7;
  if (!(b & 0x80)) goto done;
  part1 -= 0x80 << 7;
  b = *(ptr++);
  part1 += b << 14;
  if (!(b & 0x80)) goto done;
  part1 -= 0x80 << 14;
  b = *(ptr++);
  part1 += b << 21;
  if (!(b & 0x80)) goto done;
  part1 -= 0x80 << 21;
  b = *(ptr++);
  part2 = b;
  if (!(b & 0x80)) goto done;
  part2 -= 0x80;
  b = *(ptr++);
  part2 += b << 7;
  if (!(b & 0x80)) goto done;
  // "part2 -= 0x80 << 7" is irrelevant because (0x80 << 7) << 56 is 0.

  // We have overrun the maximum size of a varint (10 bytes).  Assume
  // the data is corrupt.
  *ok = false;
  return ptr;

done:
  *ok = true;
  *value = (static_cast<uint64>(part0)) | (static_cast<uint64>(part1) << 28) |
           (static_cast<uint64>(part2) << 56);
  return ptr;
}

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_PROTO_DECODE_H_
