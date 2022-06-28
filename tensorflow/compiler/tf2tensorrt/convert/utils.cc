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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "absl/strings/ascii.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace tensorrt {

string DebugString(const nvinfer1::Dims& dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  string out = StrCat("nvinfer1::Dims(nbDims=", dims.nbDims, ", d=");
  for (int i = 0; i < std::max(dims.nbDims, 0); ++i) {
    StrAppend(&out, dims.d[i]);
    StrAppend(&out, ",");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const DataType tf_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  switch (tf_type) {
    case DT_FLOAT:
      return "DT_FLOAT";
    case DT_HALF:
      return "DT_HALF";
    case DT_INT32:
      return "DT_INT32";
    case DT_INT8:
      return "DT_INT8";
    case DT_BOOL:
      return "DT_BOOL";
    default:
      return "Unknow TF DataType";
  }
}

string DebugString(const nvinfer1::DataType trt_dtype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_2(mht_2_v, 230, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      return "kFLOAT";
    case nvinfer1::DataType::kHALF:
      return "kHALF";
    case nvinfer1::DataType::kINT8:
      return "kINT8";
    case nvinfer1::DataType::kINT32:
      return "kINT32";
    case nvinfer1::DataType::kBOOL:
      return "kBOOL";
    default:
      return "Invalid TRT data type";
  }
}

string DebugString(const nvinfer1::Permutation& permutation, int len) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_3(mht_3_v, 250, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  string out = "nvinfer1::Permutation(";
  for (int i = 0; i < len; ++i) {
    StrAppend(&out, permutation.order[i], ",");
  }
  StrAppend(&out, ")");
  return out;
}

string DebugString(const ITensorProxyPtr& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_4(mht_4_v, 262, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  return StrCat(
      tensor->is_trt_tensor() ? "nvinfer1::ITensor(@" : "SimpleItensor(@",
      reinterpret_cast<uintptr_t>(&tensor), ", name=", tensor->getName(),
      ", dtype=", DebugString(tensor->getType()),
      ", dims=", DebugString(tensor->getDimensions()), ")");
}

string DebugString(const nvinfer1::ITensor& tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  return StrCat("nvinfer1::ITensor(@", reinterpret_cast<uintptr_t>(&tensor),
                ", name=", tensor.getName(),
                ", dtype=", DebugString(tensor.getType()),
                ", dims=", DebugString(tensor.getDimensions()), ")");
}

string DebugString(const std::vector<nvinfer1::Dims>& dimvec) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_6(mht_6_v, 283, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  return absl::StrCat("[",
                      absl::StrJoin(dimvec, ",",
                                    [](std::string* out, nvinfer1::Dims in) {
                                      out->append(DebugString(in));
                                    }),
                      "]");
}

string DebugString(const std::vector<TensorShape>& shapes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_7(mht_7_v, 295, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  return TensorShapeUtils::ShapeListString(shapes);
}

string DebugString(const std::vector<PartialTensorShape>& shapes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_8(mht_8_v, 302, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "DebugString");

  return PartialTensorShapeUtils::PartialShapeListString(shapes);
}

// Checks whether actual_shapes are compatible with cached_shapes. This should
// only be used in implicit batch mode (in explicit batch mode one needs to
// check the profile ranges). Therefore implicit batch mode is assumed.
// It is also assumed that both actual_shapes and cached_shapes have been
// verified by TRTEngineOp::VerifyInputShapes, which ensures that the batch size
// for all tensors are the same.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_9(mht_9_v, 316, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "AreShapesCompatible");

  auto match_shape = [](const TensorShape& actual_shape,
                        const TensorShape& cached_shape) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_10(mht_10_v, 321, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "lambda");

    // Match the rank.
    if (actual_shape.dims() != cached_shape.dims()) return false;
    // Match the batch size. In implicit batch mode cached_shape.dim_size(0) is
    // the max batch size, which can be larger than the actual batch size.
    if (actual_shape.dim_size(0) > cached_shape.dim_size(0)) return false;
    // Match remaining dimensions.
    for (int i = 1; i < actual_shape.dims(); ++i) {
      if (actual_shape.dim_size(i) != cached_shape.dim_size(i)) return false;
    }
    return true;
  };
  for (int i = 0; i < actual_shapes.size(); ++i) {
    if (!match_shape(actual_shapes[i], cached_shapes[i])) {
      return false;
    }
  }
  return true;
}
Status GetNetworkInputShapes(const nvinfer1::INetworkDefinition* network,
                             std::vector<PartialTensorShape>* input_shapes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_11(mht_11_v, 344, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "GetNetworkInputShapes");

  const int n_inputs = network->getNbInputs();
  input_shapes->resize(n_inputs);
  for (int i = 0; i < n_inputs; i++) {
    const ITensorProxyPtr input = network->getInput(i);
    TF_RETURN_IF_ERROR(DimsAdapter(input->getDimensions())
                           .PartialTensorShape(&input_shapes->at(i)));
  }
  return Status::OK();
}

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_12(mht_12_v, 358, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "TfTypeToTrtType");

  switch (tf_type) {
    case DT_FLOAT:
      *trt_type = nvinfer1::DataType::kFLOAT;
      break;
    case DT_HALF:
      *trt_type = nvinfer1::DataType::kHALF;
      break;
    case DT_INT32:
      *trt_type = nvinfer1::DataType::kINT32;
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case DT_BOOL:
      *trt_type = nvinfer1::DataType::kBOOL;
      break;
#endif
    default:
      return errors::InvalidArgument("Unsupported tensorflow data type ",
                                     DataTypeString(tf_type));
  }
  return Status::OK();
}

Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_13(mht_13_v, 384, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "TrtTypeToTfType");

  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      *tf_type = DT_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *tf_type = DT_HALF;
      break;
    case nvinfer1::DataType::kINT32:
      *tf_type = DT_INT32;
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case nvinfer1::DataType::kBOOL:
      *tf_type = DT_BOOL;
      break;
#endif
    default:
      return errors::InvalidArgument("Invalid TRT data type");
  }
  return Status::OK();
}

int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_14(mht_14_v, 409, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "GetNumberOfEngineInputs");

  int n_bindings = engine->getNbBindings();
  int n_input = 0;
  for (int i = 0; i < n_bindings; i++) {
    if (engine->bindingIsInput(i)) n_input++;
  }
  // According to TensorRT 7 doc: "If the engine has been built for K profiles,
  // the first getNbBindings() / K bindings are used by profile number 0, the
  // following getNbBindings() / K bindings are used by profile number 1 etc."
  // Therefore, to get the number of input tensors, we need to divide by the
  // the number of profiles.
  int n_profiles = engine->getNbOptimizationProfiles();
  return n_input / n_profiles;
}

absl::string_view GetDeviceName(const Node* node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTcc mht_15(mht_15_v, 427, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.cc", "GetDeviceName");

  if (node->has_assigned_device_name()) {
    return node->assigned_device_name();
  }
  return node->requested_device();
}

absl::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node) {
  absl::string_view device_name = GetDeviceName(node);
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed_name)) {
    return absl::nullopt;
  }
  return parsed_name;
}

absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a,
    const DeviceNameUtils::ParsedName& b) {
  DeviceNameUtils::ParsedName merged_name = a;
  if (!DeviceNameUtils::MergeDevNames(&merged_name, b,
                                      /*allow_soft_placement=*/false)
           .ok()) {
    return absl::nullopt;
  }
  return merged_name;
}

absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b) {
  DeviceNameUtils::ParsedName b_parsed_name;
  if (!DeviceNameUtils::ParseFullName(b, &b_parsed_name)) {
    return absl::nullopt;
  }

  return MergeIfCompatible(a, b_parsed_name);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
