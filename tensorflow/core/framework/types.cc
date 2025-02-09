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
class MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc() {
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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

struct DataTypeHasher {
  std::size_t operator()(const DataType& k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};

// Mapping from some of the DType fields, for backward compatibility. All other
// dtypes are mapped to TFT_ANY, but can be added here if a counterpart is
// defined.
auto* DT_TO_FT = new std::unordered_map<DataType, FullTypeId, DataTypeHasher>({
    {DT_FLOAT, TFT_FLOAT},
    {DT_DOUBLE, TFT_DOUBLE},
    {DT_INT32, TFT_INT32},
    {DT_UINT8, TFT_UINT8},
    {DT_INT16, TFT_INT16},
    {DT_INT8, TFT_INT8},
    {DT_STRING, TFT_STRING},
    {DT_COMPLEX64, TFT_COMPLEX64},
    {DT_INT64, TFT_INT64},
    {DT_BOOL, TFT_BOOL},
    {DT_UINT16, TFT_UINT16},
    {DT_COMPLEX128, TFT_COMPLEX128},
    {DT_HALF, TFT_HALF},
    {DT_UINT32, TFT_UINT32},
    {DT_UINT64, TFT_UINT64},
    {DT_VARIANT, TFT_LEGACY_VARIANT},
});

void map_dtype_to_tensor(const DataType& dtype, FullTypeDef& t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/framework/types.cc", "map_dtype_to_tensor");

  t.Clear();

  const auto& mapped = DT_TO_FT->find(dtype);
  // Only map known types, everything else remains unset. This is so that we
  // only set the most specific type when it is fully known. For example, if the
  // dtype is DT_VARIANT, then we don't know much and opt to assume that
  // the type is unset, rather than TFT_ANY.
  if (mapped != DT_TO_FT->end()) {
    t.set_type_id(mapped->second);
  }
}

bool DeviceType::operator<(const DeviceType& other) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/framework/types.cc", "DeviceType::operator<");

  return type_ < other.type_;
}

bool DeviceType::operator==(const DeviceType& other) const {
  return type_ == other.type_;
}

std::ostream& operator<<(std::ostream& os, const DeviceType& d) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/framework/types.cc", "operator<<");

  os << d.type();
  return os;
}

const char* const DEVICE_DEFAULT = "DEFAULT";
const char* const DEVICE_CPU = "CPU";
const char* const DEVICE_GPU = "GPU";
const char* const DEVICE_TPU = "TPU";
const char* const DEVICE_TPU_SYSTEM = "TPU_SYSTEM";

const std::string DeviceName<Eigen::ThreadPoolDevice>::value = DEVICE_CPU;
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
const std::string DeviceName<Eigen::GpuDevice>::value = DEVICE_GPU;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace {
string DataTypeStringInternal(DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_3(mht_3_v, 269, "", "./tensorflow/core/framework/types.cc", "DataTypeStringInternal");

  switch (dtype) {
    case DT_INVALID:
      return "INVALID";
    case DT_FLOAT:
      return "float";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int32";
    case DT_UINT32:
      return "uint32";
    case DT_UINT8:
      return "uint8";
    case DT_UINT16:
      return "uint16";
    case DT_INT16:
      return "int16";
    case DT_INT8:
      return "int8";
    case DT_STRING:
      return "string";
    case DT_COMPLEX64:
      return "complex64";
    case DT_COMPLEX128:
      return "complex128";
    case DT_INT64:
      return "int64";
    case DT_UINT64:
      return "uint64";
    case DT_BOOL:
      return "bool";
    case DT_QINT8:
      return "qint8";
    case DT_QUINT8:
      return "quint8";
    case DT_QUINT16:
      return "quint16";
    case DT_QINT16:
      return "qint16";
    case DT_QINT32:
      return "qint32";
    case DT_BFLOAT16:
      return "bfloat16";
    case DT_HALF:
      return "half";
    case DT_RESOURCE:
      return "resource";
    case DT_VARIANT:
      return "variant";
    default:
      LOG(ERROR) << "Unrecognized DataType enum value " << dtype;
      return strings::StrCat("unknown dtype enum (", dtype, ")");
  }
}
}  // end namespace

string DataTypeString(DataType dtype) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_4(mht_4_v, 329, "", "./tensorflow/core/framework/types.cc", "DataTypeString");

  if (IsRefType(dtype)) {
    DataType non_ref = static_cast<DataType>(dtype - kDataTypeRefOffset);
    return strings::StrCat(DataTypeStringInternal(non_ref), "_ref");
  }
  return DataTypeStringInternal(dtype);
}

bool DataTypeFromString(StringPiece sp, DataType* dt) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/framework/types.cc", "DataTypeFromString");

  if (str_util::EndsWith(sp, "_ref")) {
    sp.remove_suffix(4);
    DataType non_ref;
    if (DataTypeFromString(sp, &non_ref) && !IsRefType(non_ref)) {
      *dt = static_cast<DataType>(non_ref + kDataTypeRefOffset);
      return true;
    } else {
      return false;
    }
  }

  if (sp == "float" || sp == "float32") {
    *dt = DT_FLOAT;
    return true;
  } else if (sp == "double" || sp == "float64") {
    *dt = DT_DOUBLE;
    return true;
  } else if (sp == "int32") {
    *dt = DT_INT32;
    return true;
  } else if (sp == "uint32") {
    *dt = DT_UINT32;
    return true;
  } else if (sp == "uint8") {
    *dt = DT_UINT8;
    return true;
  } else if (sp == "uint16") {
    *dt = DT_UINT16;
    return true;
  } else if (sp == "int16") {
    *dt = DT_INT16;
    return true;
  } else if (sp == "int8") {
    *dt = DT_INT8;
    return true;
  } else if (sp == "string") {
    *dt = DT_STRING;
    return true;
  } else if (sp == "complex64") {
    *dt = DT_COMPLEX64;
    return true;
  } else if (sp == "complex128") {
    *dt = DT_COMPLEX128;
    return true;
  } else if (sp == "int64") {
    *dt = DT_INT64;
    return true;
  } else if (sp == "uint64") {
    *dt = DT_UINT64;
    return true;
  } else if (sp == "bool") {
    *dt = DT_BOOL;
    return true;
  } else if (sp == "qint8") {
    *dt = DT_QINT8;
    return true;
  } else if (sp == "quint8") {
    *dt = DT_QUINT8;
    return true;
  } else if (sp == "qint16") {
    *dt = DT_QINT16;
    return true;
  } else if (sp == "quint16") {
    *dt = DT_QUINT16;
    return true;
  } else if (sp == "qint32") {
    *dt = DT_QINT32;
    return true;
  } else if (sp == "bfloat16") {
    *dt = DT_BFLOAT16;
    return true;
  } else if (sp == "half" || sp == "float16") {
    *dt = DT_HALF;
    return true;
  } else if (sp == "resource") {
    *dt = DT_RESOURCE;
    return true;
  } else if (sp == "variant") {
    *dt = DT_VARIANT;
    return true;
  }
  return false;
}

string DeviceTypeString(const DeviceType& device_type) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_6(mht_6_v, 428, "", "./tensorflow/core/framework/types.cc", "DeviceTypeString");

  return device_type.type();
}

string DataTypeSliceString(const DataTypeSlice types) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_7(mht_7_v, 435, "", "./tensorflow/core/framework/types.cc", "DataTypeSliceString");

  string out;
  for (auto it = types.begin(); it != types.end(); ++it) {
    strings::StrAppend(&out, ((it == types.begin()) ? "" : ", "),
                       DataTypeString(*it));
  }
  return out;
}

bool DataTypeAlwaysOnHost(DataType dt) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_8(mht_8_v, 447, "", "./tensorflow/core/framework/types.cc", "DataTypeAlwaysOnHost");

  // Includes DT_STRING and DT_RESOURCE.
  switch (dt) {
    case DT_STRING:
    case DT_STRING_REF:
    case DT_RESOURCE:
      return true;
    default:
      return false;
  }
}

int DataTypeSize(DataType dt) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTcc mht_9(mht_9_v, 462, "", "./tensorflow/core/framework/types.cc", "DataTypeSize");

#define CASE(T)                  \
  case DataTypeToEnum<T>::value: \
    return sizeof(T);
  switch (dt) {
    TF_CALL_POD_TYPES(CASE);
    TF_CALL_QUANTIZED_TYPES(CASE);
    // TF_CALL_QUANTIZED_TYPES() macro does no cover quint16 and qint16, since
    // they are not supported widely, but are explicitly listed here for
    // bitcast.
    TF_CALL_qint16(CASE);
    TF_CALL_quint16(CASE);

    default:
      return 0;
  }
#undef CASE
}

// Define DataTypeToEnum<T>::value.
#define DEFINE_DATATYPETOENUM_VALUE(TYPE) \
  constexpr DataType DataTypeToEnum<TYPE>::value;

DEFINE_DATATYPETOENUM_VALUE(float);
DEFINE_DATATYPETOENUM_VALUE(double);
DEFINE_DATATYPETOENUM_VALUE(int32);
DEFINE_DATATYPETOENUM_VALUE(uint32);
DEFINE_DATATYPETOENUM_VALUE(uint16);
DEFINE_DATATYPETOENUM_VALUE(uint8);
DEFINE_DATATYPETOENUM_VALUE(int16);
DEFINE_DATATYPETOENUM_VALUE(int8);
DEFINE_DATATYPETOENUM_VALUE(tstring);
DEFINE_DATATYPETOENUM_VALUE(complex64);
DEFINE_DATATYPETOENUM_VALUE(complex128);
DEFINE_DATATYPETOENUM_VALUE(int64_t);
DEFINE_DATATYPETOENUM_VALUE(uint64);
DEFINE_DATATYPETOENUM_VALUE(bool);
DEFINE_DATATYPETOENUM_VALUE(qint8);
DEFINE_DATATYPETOENUM_VALUE(quint8);
DEFINE_DATATYPETOENUM_VALUE(qint16);
DEFINE_DATATYPETOENUM_VALUE(quint16);
DEFINE_DATATYPETOENUM_VALUE(qint32);
DEFINE_DATATYPETOENUM_VALUE(bfloat16);
DEFINE_DATATYPETOENUM_VALUE(Eigen::half);
DEFINE_DATATYPETOENUM_VALUE(ResourceHandle);
DEFINE_DATATYPETOENUM_VALUE(Variant);
#undef DEFINE_DATATYPETOENUM_VALUE

}  // namespace tensorflow
