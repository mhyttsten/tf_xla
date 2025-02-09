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
class MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc() {
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

// DecodeProto is a TensorFlow op which extracts arbitrary fields from protos
// serialized as strings.
//
// See docs in ../ops/decode_proto_op.cc.
//
// This implementation reads the serialized format using a handful of calls from
// the WireFormatLite API used by generated proto code. WireFormatLite is marked
// as an "internal" proto API but is widely used in practice and highly unlikely
// to change. This will be much faster than the previous implementation based on
// constructing a temporary dynamic message in memory and using the proto
// reflection api to read it. It can be used with any proto whose descriptors
// are available at runtime but should be competitive in speed with approaches
// that compile in the proto definitions.

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/proto/decode.h"
#include "tensorflow/core/util/proto/descriptors.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

using ::tensorflow::MakeUnique;
using ::tensorflow::protobuf::Descriptor;
using ::tensorflow::protobuf::DescriptorPool;
using ::tensorflow::protobuf::DynamicMessageFactory;
using ::tensorflow::protobuf::FieldDescriptor;
using ::tensorflow::protobuf::Message;
using ::tensorflow::protobuf::TextFormat;
using ::tensorflow::protobuf::internal::WireFormatLite;
using ::tensorflow::protobuf::io::CodedInputStream;

const bool kFailOnDecodeError = true;

// Used to store the default value of a protocol message field, casted to the
// type of the output tensor.
//
// TODO(paskin): Use absl::variant once TensorFlow gets absl dependencies.
struct DefaultValue {
  DataType dtype = DataType::DT_INVALID;
  union Value {
    bool v_bool;           // DT_BOOL
    double v_double;       // DT_DOUBLE
    float v_float;         // DT_FLOAT
    int8 v_int8;           // DT_INT8
    int32 v_int32;         // DT_INT32
    int64_t v_int64;       // DT_INT64
    const char* v_string;  // DT_STRING
    uint8 v_uint8;         // DT_UINT8
    uint8 v_uint32;        // DT_UINT32
    uint8 v_uint64;        // DT_UINT64
  };
  Value value;
};

// Initializes a DefaultValue object.  This generic template handles numeric
// types and strings are handled by a template specialization below.
//
// Args:
//   dtype: the type of the output tensor
//   value: the default value as obtained from the FieldDescriptor
//   result: the object to initialize
template <typename T>
Status InitDefaultValue(DataType dtype, const T value, DefaultValue* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_0(mht_0_v, 261, "", "./tensorflow/core/kernels/decode_proto_op.cc", "InitDefaultValue");

  result->dtype = dtype;
  switch (dtype) {
    case DT_BOOL:
      result->value.v_bool = static_cast<bool>(value);
      break;
    case DT_DOUBLE:
      result->value.v_double = static_cast<double>(value);
      break;
    case DT_FLOAT:
      result->value.v_float = static_cast<float>(value);
      break;
    case DT_INT8:
      result->value.v_int8 = static_cast<int8>(value);
      break;
    case DT_INT32:
      result->value.v_int32 = static_cast<int32>(value);
      break;
    case DT_INT64:
      result->value.v_int64 = static_cast<int64_t>(value);
      break;
    case DT_UINT8:
      result->value.v_uint8 = static_cast<uint8>(value);
      break;
    case DT_UINT32:
      result->value.v_uint32 = static_cast<uint32>(value);
      break;
    case DT_UINT64:
      result->value.v_uint64 = static_cast<uint64>(value);
      break;
    default:
      // We should never get here, given the type checking that occurs earlier.
      return errors::Internal(
          "Cannot initialize default value for unsupported type: ",
          DataTypeString(dtype));
  }
  return Status::OK();
}

template <>
Status InitDefaultValue(DataType dtype, const char* value,
                        DefaultValue* result) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_1(mht_1_v, 306, "", "./tensorflow/core/kernels/decode_proto_op.cc", "InitDefaultValue");

  // These are sanity checks that should never trigger given the code that
  // leads here.
  if (TF_PREDICT_FALSE(dtype != DT_STRING)) {
    return errors::InvalidArgument(
        "Cannot cast field to anything but DT_STRING");
  }
  if (TF_PREDICT_FALSE(value == nullptr)) {
    return errors::InvalidArgument("Null default string value.");
  }
  result->dtype = DT_STRING;
  result->value.v_string = value;
  return Status::OK();
}

// Initializes a default value from the output data type and the field
// descriptor.
Status InitDefaultValueFromFieldDescriptor(DataType dtype,
                                           const FieldDescriptor* field_desc,
                                           DefaultValue* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_2(mht_2_v, 328, "", "./tensorflow/core/kernels/decode_proto_op.cc", "InitDefaultValueFromFieldDescriptor");

  switch (field_desc->type()) {
    case WireFormatLite::TYPE_DOUBLE:
      return InitDefaultValue(dtype, field_desc->default_value_double(),
                              result);
    case WireFormatLite::TYPE_FLOAT:
      return InitDefaultValue(dtype, field_desc->default_value_float(), result);
    case WireFormatLite::TYPE_INT64:
    case WireFormatLite::TYPE_SINT64:
    case WireFormatLite::TYPE_SFIXED64:
      return InitDefaultValue(dtype, field_desc->default_value_int64(), result);
    case WireFormatLite::TYPE_FIXED64:
    case WireFormatLite::TYPE_UINT64:
      return InitDefaultValue(dtype, field_desc->default_value_uint64(),
                              result);
    case WireFormatLite::TYPE_INT32:
    case WireFormatLite::TYPE_SINT32:
    case WireFormatLite::TYPE_SFIXED32:
      return InitDefaultValue(dtype, field_desc->default_value_int32(), result);
    case WireFormatLite::TYPE_FIXED32:
    case WireFormatLite::TYPE_UINT32:
      return InitDefaultValue(dtype, field_desc->default_value_uint32(),
                              result);
    case WireFormatLite::TYPE_BOOL:
      return InitDefaultValue(dtype, field_desc->default_value_bool(), result);
    case WireFormatLite::TYPE_ENUM:
      return InitDefaultValue(dtype, field_desc->default_value_enum()->number(),
                              result);
    case WireFormatLite::TYPE_BYTES:
    case WireFormatLite::TYPE_STRING:
      // Manipulating default string values as C-style pointers should be OK
      // for typical code-generated protocol messages.  It is possible in
      // principle to register a message descriptor on the fly, and these
      // pointers may not be stable if that descriptor has a weird
      // implementation.  (But the return type of default_value_string() is
      // const string&, so it'd have to be very weird.)
      return InitDefaultValue(dtype, field_desc->default_value_string().c_str(),
                              result);
    case WireFormatLite::TYPE_GROUP:
    case WireFormatLite::TYPE_MESSAGE:
      return InitDefaultValue(dtype, "", result);
      // default: intentionally omitted in order to enable static checking.
  }
  return Status::OK();
}

// A FieldInfo holds a handful of information from the FieldDescriptor
// and user attributes.
struct FieldInfo {
  FieldInfo(const FieldDescriptor* field_desc, int user_index,
            DefaultValue def_value)
      : output_index(user_index), default_value(def_value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_3(mht_3_v, 382, "", "./tensorflow/core/kernels/decode_proto_op.cc", "FieldInfo");

    // Without this intermediate data structure, the profile had hotspots
    // calling methods of FieldDescriptor.
    number = field_desc->number();

    // The wire format library defines the same constants used in
    // descriptor.proto. This static_cast is safe because they are guaranteed to
    // stay in sync. We need the field type from the FieldDescriptor here
    // because the wire format doesn't tell us anything about what happens
    // inside a packed repeated field: there is enough information in the wire
    // format to skip the whole field but not enough to know how to parse what's
    // inside. For that we go to the schema.
    type = static_cast<WireFormatLite::FieldType>(field_desc->type());
    is_repeated = field_desc->is_repeated();
  }

  // Disable copy and move.
  FieldInfo(const FieldInfo&) = delete;
  FieldInfo& operator=(const FieldInfo&) = delete;

  // Internally we sort field descriptors by wire number for fast lookup. In
  // general this is different from the order given by the user. Output_index
  // gives the index into the field_names and output_types attributes and into
  // the output tensor list.
  int output_index = -1;

  // This is a cache of the relevant fields from `FieldDescriptorProto`. This
  // was added after noticing that FieldDescriptor->type() was using 6% of the
  // cpu profile.
  WireFormatLite::FieldType type;
  int number;
  bool is_repeated;
  DefaultValue default_value;
};

// A CountCollector counts sizes of repeated and optional fields in a proto.
//
// Each field is tracked by a single CountCollector instance. The instance
// manages a single count, which is stored as a pointer (it is intended to be a
// reference to the `sizes` output which is being filled in). The pointer is
// passed in at initialization.
//
// Counting is done as a separate pass in order to allocate output tensors all
// at once. This allows the TensorFlow runtime to optimize allocation for the
// consumer, while removing the need for copying inside this op. After this
// pass, the DenseCollector class (below) gathers the data: it is more complex
// and provides better motivation for the API here.
class CountCollector {
 public:
  CountCollector() = delete;

  // The count may be stored inside an Eigen Tensor to eliminate copying.
  explicit CountCollector(int32* count) : count_ptr_(count) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_4(mht_4_v, 437, "", "./tensorflow/core/kernels/decode_proto_op.cc", "CountCollector");
}

  // Reads (in this case counts) a single value.
  Status ReadValue(CodedInputStream* input, const FieldInfo& field) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_5(mht_5_v, 443, "", "./tensorflow/core/kernels/decode_proto_op.cc", "ReadValue");

    // Only repeated fields can have count > 1.
    if (*count_ptr_ == 0 || field.is_repeated) {
      (*count_ptr_)++;
    }
    // We expect a wire type based on the schema field_type, to allow a little
    // more checking.
    if (!SkipValue(input, field)) {
      return errors::DataLoss("ReadValue: Failed skipping field when counting");
    }
    return Status::OK();
  }

  // Reads (in this case counts) a length-delimited list of values.
  Status ReadPackedValues(CodedInputStream* input, const FieldInfo& field,
                          size_t buf_size) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_6(mht_6_v, 461, "", "./tensorflow/core/kernels/decode_proto_op.cc", "ReadPackedValues");

    if (buf_size == 0) {
      return Status::OK();
    }

    const void* tmpbuf;
    int unused_max_buf_size;

    input->GetDirectBufferPointerInline(&tmpbuf, &unused_max_buf_size);
    // This is safe because the underlying storage for the CodedInputStream is
    // owned by the input tensor. If it were a Cord or file-backed stream this
    // pointer would go stale after the bytes were skipped.
    const uint8* buf = reinterpret_cast<const uint8*>(tmpbuf);

    // Important: we skipped the input->{Push,Pop}Limit() calls for speed,
    // so the bounds check on buf_size inside Skip() is critical, and
    // must be done before scanning the contents.
    if (!input->Skip(buf_size)) {
      return errors::DataLoss("ReadPackedValues: Skipping packed field failed");
    }

    // Dispatch to the appropriately typed field reader based on the schema
    // type.
    Status st;
    switch (field.type) {
      case WireFormatLite::TYPE_DOUBLE:
        st = CountPackedFixed<double>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_FLOAT:
        st = CountPackedFixed<float>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_INT64:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_UINT64:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_INT32:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_FIXED64:
        st = CountPackedFixed<uint64>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_FIXED32:
        st = CountPackedFixed<uint32>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_BOOL:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_STRING:
        st = errors::DataLoss("TYPE_STRING encountered as packed");
        break;
      case WireFormatLite::TYPE_GROUP:
        st = errors::DataLoss("TYPE_GROUP encountered as packed");
        break;
      case WireFormatLite::TYPE_MESSAGE:
        st = errors::DataLoss("TYPE_MESSAGE encountered as packed");
        break;
      case WireFormatLite::TYPE_BYTES:
        st = errors::DataLoss("TYPE_BYTES encountered as packed");
        break;
      case WireFormatLite::TYPE_UINT32:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_ENUM:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_SFIXED32:
        st = CountPackedFixed<int32>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_SFIXED64:
        st = CountPackedFixed<int64_t>(buf, buf_size);
        break;
      case WireFormatLite::TYPE_SINT32:
        st = CountPackedVarint(buf, buf_size);
        break;
      case WireFormatLite::TYPE_SINT64:
        st = CountPackedVarint(buf, buf_size);
        break;
        // default: intentionally omitted in order to enable static checking.
    }
    if (!st.ok()) {
      return st;
    }

    if (!field.is_repeated && *count_ptr_ > 1) {
      *count_ptr_ = 1;
    }
    return Status::OK();
  }

 private:
  // Skips a length-delimited value.
  static bool SkipBytes(CodedInputStream* input) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_7(mht_7_v, 557, "", "./tensorflow/core/kernels/decode_proto_op.cc", "SkipBytes");

    uint32 length;
    if (!input->ReadVarint32(&length)) {
      return false;
    }
    return input->Skip(length);
  }

  // Counts the number of packed varints in an array. The end of a varint is
  // signaled by a value < 0x80, so counting them requires parsing the
  // bytestream. It is the caller's responsibility to ensure that len > 0.
  Status CountPackedVarint(const uint8* buf, size_t len) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_8(mht_8_v, 571, "", "./tensorflow/core/kernels/decode_proto_op.cc", "CountPackedVarint");

    const uint8* bound = buf + len;
    int count;

    // The last byte in a valid encoded varint is guaranteed to have the high
    // bit unset. We rely on this property to prevent ReadVarint64FromArray from
    // going out of bounds, so validate the end of the buf before scanning
    // anything.
    if (bound[-1] & 0x80) {
      return errors::DataLoss("Corrupt packed varint");
    }

    // Now we can trust ReadVarint64FromArray to stay in bounds.
    for (count = 0; buf < bound; ++count) {
      uint64 temp;
      bool ok;
      buf = internal::ReadVarint64FromArray(buf, &ok, &temp);
      if (!ok) {
        return errors::DataLoss("Corrupt packed varint");
      }
    }

    *count_ptr_ += count;
    return Status::OK();
  }

  // Counts the number of fixed-size values in a packed field. This can be done
  // without actually parsing anything.
  template <typename T>
  Status CountPackedFixed(const uint8* unused_buf, size_t len) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_9(mht_9_v, 603, "", "./tensorflow/core/kernels/decode_proto_op.cc", "CountPackedFixed");

    int count = len / sizeof(T);
    if (count * sizeof(T) != len) {
      return errors::DataLoss(
          "Illegal data length for packed fixed-size type: ", len);
    }
    *count_ptr_ += len / sizeof(T);
    return Status::OK();
  }

  // Skips a single value in the input stream. Dispatches to the appropriately
  // typed field skipper based on the schema type tag. This is not as permissive
  // as just handling the wire type.
  static bool SkipValue(CodedInputStream* input, const FieldInfo& field) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_10(mht_10_v, 619, "", "./tensorflow/core/kernels/decode_proto_op.cc", "SkipValue");

    uint32 tmp32;
    protobuf_uint64 tmp64;
    switch (field.type) {
      case WireFormatLite::TYPE_DOUBLE:
        return input->ReadLittleEndian64(&tmp64);
      case WireFormatLite::TYPE_FLOAT:
        return input->ReadLittleEndian32(&tmp32);
      case WireFormatLite::TYPE_INT64:
        return input->ReadVarint64(&tmp64);
      case WireFormatLite::TYPE_UINT64:
        return input->ReadVarint64(&tmp64);
      case WireFormatLite::TYPE_INT32:
        return input->ReadVarint32(&tmp32);
      case WireFormatLite::TYPE_FIXED64:
        return input->ReadLittleEndian64(&tmp64);
      case WireFormatLite::TYPE_FIXED32:
        return input->ReadLittleEndian32(&tmp32);
      case WireFormatLite::TYPE_BOOL:
        return input->ReadVarint32(&tmp32);
      case WireFormatLite::TYPE_STRING:
        return SkipBytes(input);
      case WireFormatLite::TYPE_GROUP:
        return WireFormatLite::SkipField(
            input, WireFormatLite::MakeTag(
                       field.number, WireFormatLite::WIRETYPE_START_GROUP));
      case WireFormatLite::TYPE_MESSAGE:
        return SkipBytes(input);
      case WireFormatLite::TYPE_BYTES:
        return SkipBytes(input);
      case WireFormatLite::TYPE_UINT32:
        return input->ReadVarint32(&tmp32);
      case WireFormatLite::TYPE_ENUM:
        return input->ReadVarint32(&tmp32);
      case WireFormatLite::TYPE_SFIXED32:
        return input->ReadLittleEndian32(&tmp32);
      case WireFormatLite::TYPE_SFIXED64:
        return input->ReadLittleEndian64(&tmp64);
      case WireFormatLite::TYPE_SINT32:
        return input->ReadVarint32(&tmp32);
      case WireFormatLite::TYPE_SINT64:
        return input->ReadVarint64(&tmp64);
        // default: intentionally omitted in order to enable static checking.
    }
  }

  int32* count_ptr_ = nullptr;
};

// A DenseCollector accumulates values from a proto into a tensor.
//
// There is an instance of DenseCollector for each field of each proto. The
// DenseCollector deserializes the value from the wire directly into the
// preallocated output Tensor.
//
// This class is named DenseCollector because in the future there should be a
// SparseCollector that accumulates field data into sparse tensors if the user
// requests it.
class DenseCollector {
 public:
  DenseCollector() = delete;

  // A DenseCollector applies to one field of a serialized message.
  // Note that default_value.dtype is the type of the output tensor.
  DenseCollector(uint8* datap, DefaultValue default_value, int max_repeat_count)
      : datap_(datap),
        default_value_(default_value),
        max_repeat_count_(max_repeat_count) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_11(mht_11_v, 689, "", "./tensorflow/core/kernels/decode_proto_op.cc", "DenseCollector");
}

  // Reads a value from the input stream and stores it.
  //
  // Always inlining gave a ~50% speedup on microbenchmarks at one point.
  // TODO(nix): try removing it to see if that still holds.
  // TODO(jsimsa): ABSL_ATTRIBUTE_ALWAYS_INLINE
  Status ReadValue(CodedInputStream* input, const FieldInfo& field) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_12(mht_12_v, 699, "", "./tensorflow/core/kernels/decode_proto_op.cc", "ReadValue");

    // For required and optional fields, we overwrite values[0] with
    // the latest one in the wire stream.
    // See https://developers.google.com/protocol-buffers/docs/encoding#optional
    // Only for repeated fields do we advance the next_repeat_index_ past 1.
    // TODO(nix): to handle oneof we must also zero out any previous values
    //  seen on the wire.
    int32_t index = 0;
    if (field.is_repeated) {
      index = next_repeat_index_;
    }
    next_repeat_index_ = index + 1;

    return internal::ReadValue(input, field.type, field.number,
                               default_value_.dtype, index, datap_);
  }

  // Reads and stores a length-delimited list of values.
  Status ReadPackedValues(CodedInputStream* input, const FieldInfo& field,
                          const size_t buf_size) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_13(mht_13_v, 721, "", "./tensorflow/core/kernels/decode_proto_op.cc", "ReadPackedValues");

    const void* buf;
    int unused_max_buf_size;
    input->GetDirectBufferPointerInline(&buf, &unused_max_buf_size);
    // This is safe because the underlying storage for the CodedInputStream is
    // owned by the input tensor. If it were a Cord or file-backed stream this
    // pointer would go stale after the bytes were skipped.
    if (!input->Skip(buf_size)) {
      return errors::DataLoss(
          "ReadPackedValues: Skipping packed field failed.  Field tag: ",
          field.number);
    }

    // Setting stride=0 causes new values to overwrite old ones for
    // non-repeated fields.
    const int stride = field.is_repeated ? 1 : 0;

    if (next_repeat_index_ >= max_repeat_count_) {
      return errors::DataLoss(
          "ReadPackedValues: Tried to write more entries than allowed.  "
          "Field tag: ",
          field.number, ", Max entries allowed: ", max_repeat_count_);
    } else {
      return internal::ReadPackedFromArray(buf, buf_size, field.type,
                                           field.number, default_value_.dtype,
                                           stride, &next_repeat_index_, datap_);
    }
  }

  // Fills in any missing values in the output array with defaults. Dispatches
  // to the appropriately typed field default based on the runtime type tag.
  Status FillWithDefaults() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_14(mht_14_v, 755, "", "./tensorflow/core/kernels/decode_proto_op.cc", "FillWithDefaults");

    switch (default_value_.dtype) {
      case DataType::DT_BOOL:
        return FillDefault<bool>(default_value_.value.v_bool);
      case DataType::DT_FLOAT:
        return FillDefault<float>(default_value_.value.v_float);
      case DataType::DT_DOUBLE:
        return FillDefault<double>(default_value_.value.v_double);
      case DataType::DT_INT8:
        return FillDefault<int8>(default_value_.value.v_int8);
      case DataType::DT_INT32:
        return FillDefault<int32>(default_value_.value.v_int32);
      case DataType::DT_INT64:
        return FillDefault<int64_t>(default_value_.value.v_int64);
      case DataType::DT_STRING:
        return FillDefault<tstring>(default_value_.value.v_string);
      case DataType::DT_UINT8:
        return FillDefault<uint8>(default_value_.value.v_uint8);
      case DataType::DT_UINT32:
        return FillDefault<uint32>(default_value_.value.v_uint32);
      case DataType::DT_UINT64:
        return FillDefault<uint64>(default_value_.value.v_uint64);
      default:
        // There are many tensorflow dtypes not handled here, but they
        // should not come up unless type casting is added to the Op.
        // Chaining with tf.cast() should do the right thing until then.
        return errors::DataLoss("Failed filling defaults for ",
                                DataTypeString(default_value_.dtype));
    }
  }

 private:
  // Fills empty values in the dense representation with a default value. This
  // uses next_repeat_index_ which counts the number of parsed values for the
  // field.
  template <class T>
  Status FillDefault(const T& default_value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_15(mht_15_v, 794, "", "./tensorflow/core/kernels/decode_proto_op.cc", "FillDefault");

    for (int i = next_repeat_index_; i < max_repeat_count_; i++) {
      reinterpret_cast<T*>(datap_)[i] = default_value;
    }
    return Status::OK();
  }

  int32 next_repeat_index_ = 0;

  // This is a pointer to data_[message_index_]. There is no bounds checking at
  // this level: we computed the max repeat size for each field in
  // CountCollector and use the same code to traverse it here, so we are
  // guaranteed not to be called for more items than we have allocated space.
  void* const datap_ = nullptr;

  const DefaultValue default_value_;
  const int max_repeat_count_ = 0;
};

class DecodeProtoOp : public OpKernel {
 public:
  explicit DecodeProtoOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_16(mht_16_v, 818, "", "./tensorflow/core/kernels/decode_proto_op.cc", "DecodeProtoOp");

    string descriptor_source;
    OP_REQUIRES_OK(context,
                   context->GetAttr("descriptor_source", &descriptor_source));

    // We always get back a desc_pool, but we may not own it. If we own it,
    // owned_desc_pool_ will be filled in.
    DescriptorPool const* desc_pool;
    OP_REQUIRES_OK(context, GetDescriptorPool(context->env(), descriptor_source,
                                              &desc_pool, &owned_desc_pool_));

    string message_type;
    OP_REQUIRES_OK(context, context->GetAttr("message_type", &message_type));

    const Descriptor* message_desc =
        desc_pool->FindMessageTypeByName(message_type);
    OP_REQUIRES(context, message_desc != nullptr,
                errors::InvalidArgument("No descriptor found for message type ",
                                        message_type));

    std::vector<string> field_names;
    OP_REQUIRES_OK(context, context->GetAttr("field_names", &field_names));
    std::vector<DataType> output_types;
    OP_REQUIRES_OK(context, context->GetAttr("output_types", &output_types));
    OP_REQUIRES(
        context, field_names.size() == output_types.size(),
        errors::InvalidArgument("field_names and output_types attributes must "
                                "have the same length"));

    // Gather the field descriptors and check that requested output types match.
    int field_index = 0;
    std::vector<const FieldDescriptor*> field_descs;
    std::vector<const FieldDescriptor*> exts;
    absl::flat_hash_map<string, const FieldDescriptor*> ext_name_to_field;
    std::vector<const FieldDescriptor*>::iterator ext_it = exts.begin();
    for (const string& name : field_names) {
      auto fd = message_desc->FindFieldByName(name);
      if (fd == nullptr) {
        // If field can't be found in original message, try to find a matching
        // extension (by its full_name). First check a hashmap for a matching
        // extension, and if not found, then iterate through available
        // extensions to find a match (updating the hashmap while iterating.)
        auto lookup_result = ext_name_to_field.find(name);
        if (lookup_result != ext_name_to_field.end()) {
          fd = lookup_result->second;
        } else {
          if (ext_it == exts.begin()) {
            desc_pool->FindAllExtensions(message_desc, &exts);
            ext_it = exts.begin();
          }
          while (ext_it != exts.end()) {
            auto ext_name = (*ext_it)->full_name();
            auto ext_field = *ext_it;
            ++ext_it;

            ext_name_to_field.insert({ext_name, ext_field});
            if (ext_name == name) {
              fd = ext_field;
              break;
            }
          }
        }
      }
      OP_REQUIRES(context, fd != nullptr,
                  errors::InvalidArgument("Unknown field: ", name,
                                          " in message type ", message_type));
      OP_REQUIRES(
          context,
          proto_utils::IsCompatibleType(fd->type(), output_types[field_index]),
          // Many TensorFlow types don't have corresponding proto types and the
          // user will get an error if they are requested. It would be nice to
          // allow conversions here, but tf.cast already exists so we don't
          // duplicate the functionality.
          errors::InvalidArgument("Unexpected output type for ",
                                  fd->full_name(), ": ", fd->cpp_type(), " to ",
                                  output_types[field_index]));

      field_index++;
      field_descs.push_back(fd);
    }

    // Internally we want the field_descs sorted by their number on the wire.
    // But the output tensors are allocated in the order given by the caller.
    // Build a mapping i->j, where field_descs[i] corresponds to outputs[j].
    std::vector<int> output_indices;
    output_indices.reserve(field_names.size());
    for (int i = 0; i < field_names.size(); i++) {
      output_indices.push_back(i);
    }
    std::sort(output_indices.begin(), output_indices.end(),
              [field_descs](int a, int b) {
                return field_descs[a]->number() < field_descs[b]->number();
              });

    // Now store the fields in sorted order.
    for (int i = 0; i < field_names.size(); i++) {
      const int output_index = output_indices[i];
      const DataType dtype = output_types[output_index];
      const FieldDescriptor* field_descriptor = field_descs[output_index];
      DefaultValue default_value;
      OP_REQUIRES_OK(context, InitDefaultValueFromFieldDescriptor(
                                  dtype, field_descriptor, &default_value));
      fields_.push_back(
          MakeUnique<FieldInfo>(field_descriptor, output_index, default_value));
    }

    message_prototype_ = message_factory_.GetPrototype(message_desc);
    OP_REQUIRES(context, message_prototype_ != nullptr,
                errors::InvalidArgument("Couldn't get prototype message: ",
                                        message_desc->full_name()));
    string format;
    OP_REQUIRES_OK(context, context->GetAttr("message_format", &format));
    OP_REQUIRES(
        context, format == "binary" || format == "text",
        errors::InvalidArgument("format must be one of binary or text"));
    is_binary_ = format == "binary";

    // Enable the initial protobuf sanitizer, which is much more expensive than
    // the decoder.
    // TODO(nix): Remove this once the fast decoder has passed security review.
    OP_REQUIRES_OK(context, context->GetAttr("sanitize", &sanitize_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_17(mht_17_v, 944, "", "./tensorflow/core/kernels/decode_proto_op.cc", "Compute");

    const Tensor& buf_tensor = ctx->input(0);
    int message_count = buf_tensor.NumElements();
    OP_REQUIRES(ctx, message_count >= 1,
                errors::InvalidArgument(
                    "Bufs argument must contain at least one value"));

    int field_count = fields_.size();

    // Save the argument shape for later, then flatten the input Tensor since we
    // are working componentwise. We will restore the same shape in the returned
    // Tensor.
    const TensorShape& shape_prefix = buf_tensor.shape();

    TensorShape sizes_shape = shape_prefix;
    sizes_shape.AddDim(field_count);
    Tensor* sizes_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, sizes_shape, &sizes_tensor));

    // This is used to allocate binary bufs if used. It serves only to define
    // memory ownership.
    std::vector<tstring> tmp_binary_bufs(message_count);

    // These are the actual buffers to use, which may be in tmp_binary_bufs
    // or may be pointers into the buf_tensor. Either way they are not owned
    // here.
    std::vector<const tstring*> bufs;

    if (is_binary_ && !sanitize_) {
      // Fast path.
      for (int mi = 0; mi < message_count; ++mi) {
        const tstring* buf = &buf_tensor.flat<tstring>()(mi);
        bufs.push_back(buf);
      }
    } else {
      // We will have to allocate a copy, either to convert from text to binary
      // or to sanitize a binary proto.
      for (int mi = 0; mi < message_count; ++mi) {
        ReserializeMessage(ctx, buf_tensor.flat<tstring>()(mi),
                           &tmp_binary_bufs[mi]);
        if (!ctx->status().ok()) {
          return;
        }
        bufs.push_back(&tmp_binary_bufs[mi]);
      }
    }

    // Walk through all the strings in the input tensor, counting the number of
    // fields in each. We can't allocate our actual output Tensor until we know
    // the maximum repeat count, so we do a first pass through the serialized
    // proto just counting fields. We always allocate at least one value so that
    // optional fields are populated with default values - this avoids a TF
    // conditional when handling the output data. The caller can distinguish
    // between real data and defaults using the repeat count matrix that is
    // returned by decode_proto.
    std::vector<int32> max_sizes(field_count, 1);
    for (int mi = 0; mi < message_count; ++mi) {
      CountFields(ctx, mi, *bufs[mi], sizes_tensor, &max_sizes);
      if (!ctx->status().ok()) {
        return;
      }
    }

    // Allocate the output tensors now that we've seen the max size.
    // TODO(nix): Use allocate_output_or_forward_input for the largest
    //   output tensor. This can avoid one large allocation by re-using
    //   the memory of the input tensor.
    std::vector<Tensor*> outputs(field_count);
    for (int fi = 0; fi < field_count; ++fi) {
      TensorShape flat_shape = {static_cast<int64_t>(message_count),
                                max_sizes[fi]};
      TensorShape out_shape = shape_prefix;
      out_shape.AddDim(max_sizes[fi]);

      // Surprisingly we don't specify the types from the output_types
      // attribute: that is done for us based on the Op declaration:
      //  REGISTER_OP(...)
      //    .Attr("output_types: list(type) >= 0")
      //    .Output("values: output_types")
      OP_REQUIRES_OK(ctx, ctx->allocate_output(fields_[fi]->output_index + 1,
                                               out_shape, &outputs[fi]));
    }

    // Make the second pass through the serialized proto, decoding into
    // preallocated tensors.
    AccumulateFields(ctx, bufs, outputs);
  }

 private:
  // Copy a serialized message to binary, e.g. to handle text proto inputs.
  void ReserializeMessage(OpKernelContext* ctx, const tstring& buf,
                          tstring* binary_buf) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("buf: \"" + (std::string)buf + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_18(mht_18_v, 1039, "", "./tensorflow/core/kernels/decode_proto_op.cc", "ReserializeMessage");

    // Handle text protos by translating them to binary.
    std::unique_ptr<Message> message(message_prototype_->New());
    OP_REQUIRES(ctx, message, errors::DataLoss("Initializing message failed"));

    if (is_binary_) {
      // If we get here we are sanitizing the input protobuf by parsing
      // and reserializing it with a trusted (but very slow) library.
      OP_REQUIRES(ctx, message->ParseFromString(buf),
                  errors::DataLoss("Unable to parse binary protobuf"));
    } else {
      OP_REQUIRES(ctx, TextFormat::ParseFromString(buf, message.get()),
                  errors::DataLoss("Unable to parse text protobuf"));
    }

    OP_REQUIRES(ctx, SerializeToTString(*message, binary_buf),
                errors::DataLoss("Unable to reserialize text proto as binary"));
  }

  // Count the number of occurrences of each requested field in a message batch.
  void CountFields(OpKernelContext* ctx, int message_index, const tstring& buf,
                   Tensor* sizes_tensor, std::vector<int32>* max_sizes) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("buf: \"" + (std::string)buf + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_19(mht_19_v, 1064, "", "./tensorflow/core/kernels/decode_proto_op.cc", "CountFields");

    int field_count = fields_.size();

    CodedInputStream input(reinterpret_cast<const uint8*>(buf.c_str()),
                           buf.size());

    std::vector<int32> field_sizes(field_count, 0);
    std::vector<CountCollector> counters;
    counters.reserve(field_count);
    for (int i = 0; i < field_count; i++) {
      counters.emplace_back(&field_sizes[i]);
    }

    Status st = Collect(&input, absl::MakeSpan(counters));
    if (st.ok() && !input.ConsumedEntireMessage()) {
      st = errors::DataLoss("CountFields: Failed to consume entire buffer");
    }
    if (kFailOnDecodeError) {
      OP_REQUIRES_OK(ctx, st);  // NOLINT
    }
    if (!st.ok()) {
      // This code suppresses the corrupt proto, treating it as empty
      // to avoid crashing the process.
      LOG(WARNING) << "Proto counting error for message type " << message_type_
                   << ": " << st;

      for (int fi = 0; fi < field_count; fi++) {
        field_sizes[fi] = 0;
      }
      // Finished decoding this message.
      return;
    }

    // Update the size tensor and max repeat size for each field.
    auto sizes = sizes_tensor->flat_inner_dims<int32>();
    for (int fi = 0; fi < field_count; fi++) {
      int32_t size = field_sizes[fi];
      sizes(message_index, fields_[fi]->output_index) = size;
      if ((*max_sizes)[fi] < size) {
        (*max_sizes)[fi] = size;
      }
    }
  }

  // Parse fields from a serialized message into preallocated tensors.
  void AccumulateFields(OpKernelContext* ctx,
                        const std::vector<const tstring*>& bufs,
                        std::vector<Tensor*> outputs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_20(mht_20_v, 1114, "", "./tensorflow/core/kernels/decode_proto_op.cc", "AccumulateFields");

    struct TensorInfo {
      explicit TensorInfo(Tensor* tensor) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_21(mht_21_v, 1119, "", "./tensorflow/core/kernels/decode_proto_op.cc", "TensorInfo");

        // Note that we can decode only max_repeat_count values before overflow.
        // No other bounds checking is done for repeated fields. For
        // optional fields there is a check to make sure that only the last
        // value on the wire appears in the output tensor.
        dtype = tensor->dtype();
        last_dim_size = tensor->dim_size(tensor->dims() - 1);

        if (dtype != DT_STRING) {
          const int element_size = DataTypeSize(dtype);
          CHECK_GT(element_size, 0);
          stride = last_dim_size * element_size;

          const int64_t flatshape[1] = {tensor->NumElements() * element_size};
          data = tensor->bit_casted_shaped<uint8, 1>(flatshape).data();
        } else {
          // DataTypeSize() returns 0 for string types.
          stride = last_dim_size * sizeof(tstring);
          data = reinterpret_cast<uint8*>(tensor->flat<tstring>().data());
        }
      }

      DataType dtype;
      int last_dim_size;
      int stride;
      uint8* data;
    };

    int field_count = fields_.size();

    std::vector<TensorInfo> tensors;
    tensors.reserve(field_count);
    for (int fi = 0; fi < field_count; fi++) {
      tensors.emplace_back(outputs[fi]);
    }

    for (int message_index = 0; message_index < bufs.size(); ++message_index) {
      const tstring& buf = *bufs[message_index];

      std::vector<DenseCollector> collectors;
      collectors.reserve(field_count);
      for (int output_index = 0; output_index < field_count; ++output_index) {
        const TensorInfo& info = tensors[output_index];
        const FieldInfo* field_info = fields_[output_index].get();
        DCHECK(field_info != nullptr);
        const DefaultValue default_value = field_info->default_value;
        collectors.emplace_back(info.data + message_index * info.stride,
                                default_value, info.last_dim_size);
      }

      // Fill in output tensors from the wire.
      CodedInputStream input(reinterpret_cast<const uint8*>(buf.c_str()),
                             buf.size());
      Status st = Collect(&input, absl::MakeSpan(collectors));
      if (st.ok() && !input.ConsumedEntireMessage()) {
        st = errors::DataLoss(
            "AccumulateFields: Failed to consume entire buffer");
      }
      if (kFailOnDecodeError) {
        OP_REQUIRES_OK(ctx, st);  // NOLINT
      }
      if (!st.ok()) {
        // This code suppresses the corrupt proto, treating it as empty
        // to avoid crashing training.
        LOG(WARNING) << "Proto counting error for message type "
                     << message_type_ << ": " << st;
      }

      // Fill the remainder of the dense outputs with default values.
      for (auto& collector : collectors) {
        OP_REQUIRES_OK(ctx, collector.FillWithDefaults());
      }
    }
  }

  // Traverses a serialized protobuf, dispatching values to the collectors.
  template <class CollectorClass>
  Status Collect(CodedInputStream* input,
                 absl::Span<CollectorClass> collectors) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_22(mht_22_v, 1200, "", "./tensorflow/core/kernels/decode_proto_op.cc", "Collect");

    // At the beginning of each loop, the last field number that was seen,
    // regardless of whether it was collected or not, or -1 if no field has
    // been seen before.
    int last_seen_field_number = -1;
    // The FieldInfo that is expected to be used next.
    // It was either used to collect the last seen field number, or if the
    // last seen field number was not in fields_, it is the next FieldInfo after
    // the last seen field number. At the beginning it is the first FieldInfo.
    auto expected_field_info_iter = fields_.begin();

    // The 'tag' variable should always be treated as tainted.
    for (uint32 tag = input->ReadTag();
         tag != 0 && WireFormatLite::GetTagWireType(tag) !=
                         WireFormatLite::WIRETYPE_END_GROUP;
         tag = input->ReadTag()) {
      DCHECK(expected_field_info_iter == fields_.begin() ||
             last_seen_field_number >
                 (*(expected_field_info_iter - 1))->number);
      DCHECK(expected_field_info_iter == fields_.end() ||
             last_seen_field_number <= (*expected_field_info_iter)->number);

      // The field wire number.
      const int field_number = WireFormatLite::GetTagFieldNumber(tag);
      // The field info associated with the field wire number.
      const FieldInfo* field_info = nullptr;

      // fields_ are ordered by their field numbers. If the field numbers
      // on wire are also ordered (which is a convention), then we can
      // monotonically increment `expected_field_info_iter` as the field
      // numbers on wire get larger. If we detect any out-of-order
      // field number, we reset `expected_field_info_iter`, and expect that
      // future wire numbers are ordered. This algorithm is quadratic in the
      // worst case where field numbers on wire are in descending order, however
      // it works well in the case where two serialized protobufs are
      // concatenated together.
      if (field_number < last_seen_field_number) {
        expected_field_info_iter = fields_.begin();
      }

      // Advance expected_field_info_iter until
      // field_number <= expected_field_number.
      for (; expected_field_info_iter != fields_.end();
           ++expected_field_info_iter) {
        DCHECK(expected_field_info_iter == fields_.begin() ||
               field_number > (*(expected_field_info_iter - 1))->number);
        const FieldInfo* expected_field_info = expected_field_info_iter->get();
        if (field_number <= expected_field_info->number) {
          if (field_number == expected_field_info->number) {
            field_info = expected_field_info;
          }
          break;
        }
      }
      last_seen_field_number = field_number;
      if (!field_info) {
        // This DCHECK verifies that if we skip a field, we didn't want it.
        // In particular, field_builders is empty or the field_number is either:
        // before fields_.begin().number or  after (fields_.end() - 1).number or
        // in-between expected_field_info_iter and expected_field_info_iter - 1.
        DCHECK(fields_.empty() || (field_number < (*fields_.begin())->number) ||
               (field_number > (*(fields_.end() - 1))->number) ||
               (((*(expected_field_info_iter - 1))->number < field_number) &&
                (field_number < (*(expected_field_info_iter))->number)));
        // Unknown and unrequested fields are skipped.
        if (!WireFormatLite::SkipField(input, tag)) {
          return errors::DataLoss("Failed skipping unrequested field");
        }
        continue;
      }

      TF_RETURN_IF_ERROR(CollectField(
          *field_info, WireFormatLite::GetTagWireType(tag), input,
          &collectors[expected_field_info_iter - fields_.begin()]));
    }
    return Status::OK();
  }

  // Collects values for a single field.
  template <class CollectorClass>
  Status CollectField(const FieldInfo& field,
                      WireFormatLite::WireType wire_type,
                      CodedInputStream* input, CollectorClass* collector) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdecode_proto_opDTcc mht_23(mht_23_v, 1285, "", "./tensorflow/core/kernels/decode_proto_op.cc", "CollectField");

    // The wire format library defines the same constants used in
    // descriptor.proto. This static_cast is safe because they are guaranteed to
    // stay in sync.
    //
    // We need the field type from the FieldDescriptor here because the wire
    // format doesn't tell us anything about what happens inside a packed
    // repeated field: there is enough information in the wire format to skip
    // the whole field but not enough to know how to parse what's inside. For
    // that we go to the schema.
    WireFormatLite::WireType schema_wire_type =
        WireFormatLite::WireTypeForFieldType(field.type);

    // Handle packed repeated fields. SkipField would skip the whole
    // length-delimited blob without letting us count the values, so we have to
    // scan them ourselves.
    if (wire_type == WireFormatLite::WIRETYPE_LENGTH_DELIMITED &&
        schema_wire_type != WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
      // Handle packed repeated primitives.
      int length;
      if (!input->ReadVarintSizeAsInt(&length)) {
        return errors::DataLoss("CollectField: Failed reading packed size");
      }
      return collector->ReadPackedValues(input, field, length);
    }

    // Read ordinary values, including strings, bytes, and messages.
    if (wire_type != schema_wire_type) {
      if (!WireFormatLite::SkipField(
              input, WireFormatLite::MakeTag(field.number, wire_type))) {
        return errors::DataLoss(
            "CollectField: Failed skipping malformed field");
      }
      return Status::OK();
    }
    return collector->ReadValue(input, field);
  }

  string message_type_;
  // Note that fields are sorted by increasing field number, which is not in
  // general the order given by the user-specified field_names and output_types
  // Op attributes.
  std::vector<std::unique_ptr<const FieldInfo>> fields_;

  // Owned_desc_pool_ is null when using descriptor_source=local.
  std::unique_ptr<DescriptorPool> owned_desc_pool_;
  DynamicMessageFactory message_factory_;
  const Message* message_prototype_;

  // True if decoding binary format, false if decoding text format.
  bool is_binary_;

  // True if the protos should be sanitized before parsing. Enables the initial
  // protobuf sanitizer, which is much more expensive than the decoder. The flag
  // defaults to true but can be set to false for trusted sources.
  //
  // TODO(nix): Flip the default to false when the fast decoder has passed
  // security review.
  bool sanitize_;

  TF_DISALLOW_COPY_AND_ASSIGN(DecodeProtoOp);
};

REGISTER_KERNEL_BUILDER(Name("DecodeProtoV2").Device(DEVICE_CPU),
                        DecodeProtoOp);

}  // namespace
}  // namespace tensorflow
