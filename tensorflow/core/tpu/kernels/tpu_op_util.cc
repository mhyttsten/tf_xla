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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {
namespace {
std::string CreateShapePrefix(
    const std::vector<tensorflow::TensorShape>& dynamic_shapes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/tpu/kernels/tpu_op_util.cc", "CreateShapePrefix");

  std::string shapes_prefix;
  for (const TensorShape& shape : dynamic_shapes) {
    for (int64_t size : shape.dim_sizes()) {
      absl::StrAppend(&shapes_prefix, size, ",");
    }
    absl::StrAppend(&shapes_prefix, ";");
  }
  return shapes_prefix;
}

// Include compilation configurations of the arguments that are not captured
// by the called graph.
std::string CreateConfigPrefix(const TPUCompileMetadataProto& metadata) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/tpu/kernels/tpu_op_util.cc", "CreateConfigPrefix");

  std::string config_prefix;
  for (const auto& arg : metadata.args()) {
    if (arg.is_same_data_across_replicas()) {
      absl::StrAppend(&config_prefix, ":s");
      // Same.
    } else {
      // Different.
      absl::StrAppend(&config_prefix, ":");
    }
    if (arg.enable_xla_sharding() ==
        tpu::TPUCompileMetadataProto::Arg::ALLOWED) {
      // Enabled.
      absl::StrAppend(&config_prefix, "e");
    }
    if (arg.unrestricted_layout()) {
      // Unrestricted.
      absl::StrAppend(&config_prefix, ":u");
    }
    absl::StrAppend(&config_prefix, ",type(", arg.dtype(), ")");
    if (arg.has_shape()) {
      absl::StrAppend(&config_prefix, ",shape(");
      for (const auto& dim : arg.shape().dim()) {
        absl::StrAppend(&config_prefix, dim.size(), ",");
      }
      absl::StrAppend(&config_prefix, ")");
    }
  }
  return config_prefix;
}
}  // namespace

uint64 CreateFingerprintWithNameAndShapes(
    uint64 name, const std::vector<tensorflow::TensorShape>& shapes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/tpu/kernels/tpu_op_util.cc", "CreateFingerprintWithNameAndShapes");

  std::string shape_prefix = CreateShapePrefix(shapes);
  VLOG(2) << "CreateFingerprintWithNameAndShapes, name: " << name
          << ", shape_prefix: " << shape_prefix;
  return TpuCompileInterface::Get()->FingerprintString(
      absl::StrCat(name, "_", shape_prefix));
}

// Return fingerprint_in_metadata if it's not empty; otherwise read input tensor
// data to compute the fingerprint.
std::string GuaranteedConstFingerprint(
    const string& fingerprint_in_metadata,
    const OpInputList& guaranteed_constants) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("fingerprint_in_metadata: \"" + fingerprint_in_metadata + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/tpu/kernels/tpu_op_util.cc", "GuaranteedConstFingerprint");

  if (fingerprint_in_metadata.empty()) {
    uint64_t fingerprint = 0;
    for (const Tensor& constant : guaranteed_constants) {
      fingerprint =
          tpu::OpsApiFn()->TpuCompile_CreateGuaranteedConstFingerprintFn(
              fingerprint, constant.tensor_data().data(),
              constant.tensor_data().size());
    }
    return std::to_string(fingerprint);
  } else {
    return fingerprint_in_metadata;
  }
}

// The `guaranteed_constants` must be passed as reference due to the lazy
// evaluation of `guaranteed_const_fingerprint()` callback.
TpuCompilationCacheKey CreateCompilationCacheKey(
    absl::string_view function_name, uint64 function_library_fingerprint,
    uint64 mlir_module_fingerprint, const OpInputList& guaranteed_constants,
    const std::vector<TensorShape>& dynamic_shapes,
    const TPUCompileMetadataProto& metadata,
    const TpuMeshStateInterface& mesh_state, uint64_t session_id,
    ResourceMgr* resource_mgr) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("function_name: \"" + std::string(function_name.data(), function_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_op_utilDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/tpu/kernels/tpu_op_util.cc", "CreateCompilationCacheKey");

  VLOG(1) << "FunctionLibraryFingerprint:" << function_library_fingerprint;
  std::string shapes_prefix = CreateShapePrefix(dynamic_shapes);
  VLOG(1) << "shapes_prefix = " << shapes_prefix;
  std::string config_prefix = CreateConfigPrefix(metadata);
  VLOG(1) << "config_prefix = " << config_prefix;
  std::vector<int32_t> flattened_device_ids;
  if (metadata.has_device_assignment()) {
    for (const auto& device :
         metadata.device_assignment().computation_devices()) {
      flattened_device_ids.insert(flattened_device_ids.end(),
                                  device.replica_device_ids().begin(),
                                  device.replica_device_ids().end());
    }
  }
  CompilationCacheKeyResult result =
      tpu::OpsApiFn()->TpuCompile_CreateCompilationCacheKeyFn(
          CompilationCacheKeyProperty{
              config_prefix.data(), shapes_prefix.data(), function_name.data(),
              mlir_module_fingerprint, flattened_device_ids.data(),
              flattened_device_ids.size(), guaranteed_constants.size(),
              function_library_fingerprint, metadata.num_cores_per_replica(),
              metadata.num_replicas(), mesh_state.data(), session_id,
              resource_mgr});
  auto buffer_cleanup = gtl::MakeCleanup([result]() {
    tpu::OpsApiFn()->TpuCompile_DestroyCompilationCacheKeyFn(result);
  });
  TpuCompilationCacheKey key;
  key.prefix = result.key;
  key.debug_string = result.debug_string;
  key.session_id = session_id;

  // Guaranteed constants can be different across sessions. Use session_handle
  // and guaranteed_const fingerprint to guarantee no collision.
  if (guaranteed_constants.size() > 0) {
    key.has_guaranteed_const = true;
    key.session_handle = metadata.session_handle();
    // Both `metadata` and `guaranteed_constants` lifetime are captured by
    // reference based on the assumption that these variables lifetime is
    // managed through the `TPUCompileOpKernelImpl` that outlives the
    // lifetime of the compilation cache lookups.
    string fingerprint;
    key.guaranteed_const_fingerprint = [&metadata, &guaranteed_constants,
                                        fingerprint]() mutable {
      if (fingerprint.empty()) {
        fingerprint = GuaranteedConstFingerprint(
            metadata.guaranteed_const_fingerprint(), guaranteed_constants);
      }
      return fingerprint;
    };
  }
  return key;
}
}  // namespace tpu
}  // namespace tensorflow
