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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_util.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/tpu/tpu_api.h"

namespace tensorflow {
namespace tpu {

std::string SessionNameFromMetadata(const SessionMetadata* session_metadata) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "SessionNameFromMetadata");

  return session_metadata ? session_metadata->name() : "";
}

std::string ProtoKeyForComputation(const std::string& key, int core) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "ProtoKeyForComputation");

  return absl::StrCat(key, ":", core);
}

xla::StatusOr<TpuCompilationCacheKey> ParseCompilationCacheKey(
    const std::string& key) {
  const std::vector<std::string> splits = absl::StrSplit(key, '|');
  if (splits.size() == 1) {
    // No guaranteed_const.
    return TpuCompilationCacheKey(key);
  } else if (splits.size() != 3) {
    return errors::InvalidArgument("Invalid TPU compilation cache key:", key);
  }

  TpuCompilationCacheKey parsed_key(splits.at(0));
  parsed_key.has_guaranteed_const = true;
  parsed_key.session_handle = splits.at(1);
  const string fingerprint = splits.at(2);
  parsed_key.guaranteed_const_fingerprint = [fingerprint] {
    return fingerprint;
  };
  return parsed_key;
}

xla::CompileOnlyClient::AotXlaComputationInstance
BuildAotXlaComputationInstance(
    const XlaCompiler::CompilationResult& compilation_result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "BuildAotXlaComputationInstance");

  xla::CompileOnlyClient::AotXlaComputationInstance instance;
  instance.computation = compilation_result.computation.get();
  for (const xla::Shape& shape : compilation_result.xla_input_shapes) {
    instance.argument_layouts.push_back(&shape);
  }
  instance.result_layout = &compilation_result.xla_output_shape;
  return instance;
}

Status ShapeTensorToTensorShape(const Tensor& tensor, TensorShape* shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "ShapeTensorToTensorShape");

  if (tensor.dtype() != DT_INT64 ||
      !TensorShapeUtils::IsVector(tensor.shape())) {
    return errors::InvalidArgument("Shape tensor must be an int64 vector.");
  }
  const int64_t rank = tensor.NumElements();
  auto tensor_dims = tensor.flat<int64_t>();
  std::vector<int64_t> dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dims[i] = tensor_dims(i);
  }
  return TensorShapeUtils::MakeShape(dims, shape);
}

Status DynamicShapesToTensorShapes(const OpInputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "DynamicShapesToTensorShapes");

  shapes->resize(dynamic_shapes.size());
  for (int i = 0; i < dynamic_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(
        ShapeTensorToTensorShape(dynamic_shapes[i], &(*shapes)[i]));
  }
  return Status::OK();
}

Status DynamicShapesToTensorShapes(const InputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_utilDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/tpu/kernels/tpu_util.cc", "DynamicShapesToTensorShapes");

  shapes->resize(dynamic_shapes.end() - dynamic_shapes.begin());
  size_t i = 0;
  for (auto& dynamic_shape : dynamic_shapes) {
    TF_RETURN_IF_ERROR(
        ShapeTensorToTensorShape(dynamic_shape.tensor(), &(*shapes)[i]));
    ++i;
  }
  return Status::OK();
}

xla::StatusOr<std::unique_ptr<::grpc::ServerBuilder>> CreateServerBuilder(
    int serving_port) {
  auto server_builder = absl::make_unique<::grpc::ServerBuilder>();
  server_builder->AddListeningPort(
      absl::StrFormat("[::]:%d", serving_port),
      ::grpc::InsecureServerCredentials());  // NOLINT
  return std::move(server_builder);
}
}  // namespace tpu
}  // namespace tensorflow
