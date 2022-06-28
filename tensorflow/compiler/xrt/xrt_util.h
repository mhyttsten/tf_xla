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

// Utility functions in support of the XRT API.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
#define TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_memory_manager.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/compiler/xrt/xrt_state.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Factory class which creates NCCL unique IDs based on the replicas
// participating to a given communication. This is only used for GPU backends.
struct NcclUniqueIdFactory {
  virtual ~NcclUniqueIdFactory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh mht_0(mht_0_v, 212, "", "./tensorflow/compiler/xrt/xrt_util.h", "~NcclUniqueIdFactory");
}

  // Generates the NCCL unique ID for the given set of replica IDs.
  virtual std::string GetUniqueId(absl::Span<const int64_t> replicas) = 0;
};

void SetNcclUniqueIdFactory(std::shared_ptr<NcclUniqueIdFactory> factory);

std::shared_ptr<NcclUniqueIdFactory> GetNcclUniqueIdFactory();

struct InputCoords {
  explicit InputCoords(int64_t handle) : handle(handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh mht_1(mht_1_v, 226, "", "./tensorflow/compiler/xrt/xrt_util.h", "InputCoords");
}
  InputCoords(int64_t handle, xla::ShapeIndex index)
      : handle(handle), index(std::move(index)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTh mht_2(mht_2_v, 231, "", "./tensorflow/compiler/xrt/xrt_util.h", "InputCoords");
}

  int64_t handle = 0;
  xla::ShapeIndex index;
};

// Filters the debug options provided as argument according to the value of the
// TF_XLA_DEBUG_OPTIONS_PASSTHROUGH environment variable. If such variable is
// set to "1" or "true", the debug options will be returned as is. Otherwise
// only a subset of them will be set in the returned ones, and all the paths
// contained in it, will be limited to gs:// and bigstore:// ones.
xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options);

// Populates the input_coords with a list of input coordinates from a input_name
// op argument.
xla::StatusOr<std::vector<InputCoords>> GetComputationInputs(
    OpKernelContext* context, const char* input_name);

bool InputShapeMatches(const xla::Shape& parameter_shape,
                       const xla::Shape& input_shape);

xla::StatusOr<std::vector<RefPtr<XRTTupleAllocation>>> GetInputTupleAllocations(
    const std::vector<InputCoords>& input_coords,
    XRTMemoryManager::WorkingSet* working_set, xla::Backend* backend,
    int64_t num_input_shapes,
    const std::function<xla::Shape(int64_t)>& shape_getter, bool release_inputs,
    se::DeviceMemoryAllocator* allocator);

Status RebuildOutputAliases(
    const RefPtr<XRTTupleAllocation>& output_tuple,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const xla::HloInputOutputAliasConfig& input_output_alias);

xla::StatusOr<std::vector<xla::ExecutionInput>> GetArgumentsBuffers(
    const xla::HloInputOutputAliasConfig& input_output_alias,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const std::vector<bool>& input_is_dynamic, bool release_inputs);

// Create the XRT execute output tensor given the computation result
// (output_tuple). The return_exploded_tuple tells whether a tuple result should
// be returned as vector of handles representing each tuple child.
Status CreateExecuteOutput(OpKernelContext* context,
                           XRTMemoryManager* memory_manager,
                           RefPtr<XRTTupleAllocation> output_tuple,
                           bool return_exploded_tuple);

// Drives the XRT chained computation execution given the supplied core execute
// function.
using ChainedExecuteFn =
    std::function<xla::StatusOr<RefPtr<XRTTupleAllocation>>(
        const xrt::XRTChainedExecuteOp&,
        absl::Span<const RefPtr<XRTTupleAllocation>>)>;
Status ExecuteChained(OpKernelContext* context,
                      const RefPtr<XRTMemoryManager>& memory_manager,
                      xla::Backend* backend, int device_ordinal,
                      const xrt::XRTChainedExecutePlan& plan,
                      const xrt::XRTChainedExecuteConfig& config,
                      const ChainedExecuteFn& execute_op,
                      se::DeviceMemoryAllocator* allocator);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_UTIL_H_
