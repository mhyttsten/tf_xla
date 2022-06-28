/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Utilities for saving/restoring tensor slice checkpoints.

#ifndef TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh() {
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


#include <string>  // for string
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"  // for Status
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

namespace checkpoint {

// The key for the metadata in the tensor slice checkpoint files. It is "" so
// that the metadata is always at the beginning of a checkpoint file.
extern const char kSavedTensorSlicesKey[];

// Encode a tensor name + a tensor slice into an ordered code and outputs it as
// a string.
// The format is
//  <0>
//  <tensor_name>
//  <rank>
//  <dim-0-start><dim-0-length>
//  <dim-1-start><dim-1-length>
//  ...

string EncodeTensorNameSlice(const string& name,
                             const tensorflow::TensorSlice& slice);

// Parse out the name and the slice from string encoded as an ordered code.
Status DecodeTensorNameSlice(const string& code, string* name,
                             tensorflow::TensorSlice* slice);

// Extracts the full shape, slice spec, and shape of the slice from
// "shape_and_slice".  On non-OK return, caller must clear the out-arguments
// before reusing.
Status ParseShapeAndSlice(const string& shape_and_slice, TensorShape* shape,
                          TensorSlice* slice, TensorShape* shape_slice);

template <typename T>
struct SaveTypeTraits;

template <typename T>
int TensorProtoDataSize(const TensorProto& t);

template <typename T>
const typename SaveTypeTraits<T>::SavedType* TensorProtoData(
    const TensorProto& t);

template <typename T>
typename SaveTypeTraits<T>::RepeatedField* MutableTensorProtoData(
    TensorProto* t);

template <typename T>
void Fill(T* data, size_t n, TensorProto* t);

#define TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, STYPE)      \
  template <>                                                            \
  struct SaveTypeTraits<TYPE> {                                          \
    static constexpr bool supported = true;                              \
    typedef STYPE SavedType;                                             \
    typedef protobuf::RepeatedField<FTYPE> RepeatedField;                \
  };                                                                     \
  template <>                                                            \
  inline const STYPE* TensorProtoData<TYPE>(const TensorProto& t) {      \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Restore"); \
    return reinterpret_cast<const STYPE*>(t.FIELD##_val().data());       \
  }                                                                      \
  template <>                                                            \
  inline protobuf::RepeatedField<FTYPE>* MutableTensorProtoData<TYPE>(   \
      TensorProto * t) {                                                 \
    static_assert(SaveTypeTraits<TYPE>::supported,                       \
                  "Specified type " #TYPE " not supported for Save");    \
    return reinterpret_cast<protobuf::RepeatedField<FTYPE>*>(            \
        t->mutable_##FIELD##_val());                                     \
  }

#define TENSOR_PROTO_EXTRACT_TYPE(TYPE, FIELD, FTYPE)             \
  TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, FTYPE)     \
  template <>                                                     \
  inline int TensorProtoDataSize<TYPE>(const TensorProto& t) {    \
    return t.FIELD##_val_size();                                  \
  }                                                               \
  template <>                                                     \
  inline void Fill(const TYPE* data, size_t n, TensorProto* t) {  \
    typename protobuf::RepeatedField<FTYPE> copy(data, data + n); \
    t->mutable_##FIELD##_val()->Swap(&copy);                      \
  }

// Complex needs special treatment since proto doesn't have native complex
#define TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(TYPE, FIELD, FTYPE)       \
  TENSOR_PROTO_EXTRACT_TYPE_HELPER(TYPE, FIELD, FTYPE, TYPE)        \
  template <>                                                       \
  inline int TensorProtoDataSize<TYPE>(const TensorProto& t) {      \
    return t.FIELD##_val_size() / 2;                                \
  }                                                                 \
  template <>                                                       \
  inline void Fill(const TYPE* data, size_t n, TensorProto* t) {    \
    const FTYPE* sub = reinterpret_cast<const FTYPE*>(data);        \
    typename protobuf::RepeatedField<FTYPE> copy(sub, sub + 2 * n); \
    t->mutable_##FIELD##_val()->Swap(&copy);                        \
  }

TENSOR_PROTO_EXTRACT_TYPE(bool, bool, bool);
TENSOR_PROTO_EXTRACT_TYPE(float, float, float);
TENSOR_PROTO_EXTRACT_TYPE(double, double, double);
TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(complex64, scomplex, float);
TENSOR_PROTO_EXTRACT_TYPE_COMPLEX(complex128, dcomplex, double);
TENSOR_PROTO_EXTRACT_TYPE(int32, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(uint32, uint32, uint32);
TENSOR_PROTO_EXTRACT_TYPE(int64_t, int64, protobuf_int64);
TENSOR_PROTO_EXTRACT_TYPE(uint64, uint64, protobuf_uint64);
TENSOR_PROTO_EXTRACT_TYPE(uint16, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(uint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(int16, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(qint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(quint8, int, int32);
TENSOR_PROTO_EXTRACT_TYPE(quint16, int, int32);

#undef TENSOR_PROTO_EXTRACT_TYPE_COMPLEX
#undef TENSOR_PROTO_EXTRACT_TYPE_HELPER
#undef TENSOR_PROTO_EXTRACT_TYPE

// Custom implementation for qint32, based on the one for int32.

template <>
struct SaveTypeTraits<qint32> : SaveTypeTraits<int32> {};

template <>
inline int TensorProtoDataSize<qint32>(const TensorProto& t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_0(mht_0_v, 320, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoDataSize<qint32>");

  return t.int_val_size();
}

template <>
inline const int32* TensorProtoData<qint32>(const TensorProto& t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_1(mht_1_v, 328, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoData<qint32>");

  static_assert(SaveTypeTraits<qint32>::supported,
                "Specified type qint32 not supported for Restore");
  return reinterpret_cast<const int32*>(t.int_val().data());
}

inline void Fill(const qint32* data, size_t n, TensorProto* t) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_2(mht_2_v, 337, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "Fill");

  const int32* p = reinterpret_cast<const int32*>(data);
  typename protobuf::RepeatedField<int32> copy(p, p + n);
  t->mutable_int_val()->Swap(&copy);
}

// Custom implementation for Eigen::half.

template <>
struct SaveTypeTraits<Eigen::half> {
  static constexpr bool supported = true;
  typedef int SavedType;
  typedef protobuf::RepeatedField<int32> RepeatedField;
};

template <>
inline int TensorProtoDataSize<Eigen::half>(const TensorProto& t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_3(mht_3_v, 356, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoDataSize<Eigen::half>");

  return t.half_val_size();
}

template <>
inline const int* TensorProtoData<Eigen::half>(const TensorProto& t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_4(mht_4_v, 364, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoData<Eigen::half>");

  return t.half_val().data();
}

template <>
inline protobuf::RepeatedField<int32>* MutableTensorProtoData<Eigen::half>(
    TensorProto* t) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_5(mht_5_v, 373, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "MutableTensorProtoData<Eigen::half>");

  return t->mutable_half_val();
}

template <>
inline void Fill(const Eigen::half* data, size_t n, TensorProto* t) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_6(mht_6_v, 381, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "Fill");

  typename protobuf::RepeatedField<int32>* val = t->mutable_half_val();
  val->Resize(n, 0);
  for (size_t i = 0; i < n; ++i) {
    val->Set(i, Eigen::numext::bit_cast<uint16>(data[i]));
  }
}

// Custom implementation for string.

template <>
struct SaveTypeTraits<tstring> {
  static constexpr bool supported = true;
  typedef const string* SavedType;
  typedef protobuf::RepeatedPtrField<string> RepeatedField;
};

template <>
inline int TensorProtoDataSize<tstring>(const TensorProto& t) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_7(mht_7_v, 402, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoDataSize<tstring>");

  return t.string_val_size();
}

template <>
inline const string* const* TensorProtoData<tstring>(const TensorProto& t) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_8(mht_8_v, 410, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "TensorProtoData<tstring>");

  static_assert(SaveTypeTraits<tstring>::supported,
                "Specified type tstring not supported for Restore");
  return t.string_val().data();
}

template <>
inline protobuf::RepeatedPtrField<string>* MutableTensorProtoData<tstring>(
    TensorProto* t) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_9(mht_9_v, 421, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "MutableTensorProtoData<tstring>");

  static_assert(SaveTypeTraits<tstring>::supported,
                "Specified type tstring not supported for Save");
  return t->mutable_string_val();
}

template <>
inline void Fill(const tstring* data, size_t n, TensorProto* t) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSsaved_tensor_slice_utilDTh mht_10(mht_10_v, 431, "", "./tensorflow/core/util/saved_tensor_slice_util.h", "Fill");

  typename protobuf::RepeatedPtrField<string> copy(data, data + n);
  t->mutable_string_val()->Swap(&copy);
}

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SAVED_TENSOR_SLICE_UTIL_H_
