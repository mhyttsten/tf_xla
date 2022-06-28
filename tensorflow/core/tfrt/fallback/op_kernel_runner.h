/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
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
class MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh {
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
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh() {
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


#include <assert.h>
#include <stddef.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tfrt_stub {

class OpKernelRunner {
 public:
  static StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view device_name, int num_args,
      const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
      const tensorflow::DeviceMgr& device_manager,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime);

  static StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, int num_args,
      const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime,
      tensorflow::Device* device);

  OpKernelRunner() = default;

  explicit operator bool() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "bool");
 return op_kernel_ != nullptr; }

  void Run(OpKernelContext* context) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "Run");

    DVLOG(1) << "KernelFallbackExecuteCompat Running Op: "
             << op_kernel_->def().DebugString()
             << ", on Device: " << device_->name();

    op_kernel_->Compute(context);
  }

  void RunAsync(OpKernelContext* context,
                AsyncOpKernel::DoneCallback done_callback) const;

  bool IsAsync() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "IsAsync");
 return is_async_; }

  tensorflow::OpKernel* op_kernel() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_3(mht_3_v, 251, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "op_kernel");
 return op_kernel_.get(); }
  tensorflow::Device* device() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_4(mht_4_v, 255, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "device");
 return device_; }
  tensorflow::FunctionLibraryRuntime* function_library_runtime() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_5(mht_5_v, 259, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "function_library_runtime");

    return function_library_runtime_;
  }
  tensorflow::ResourceMgr* resource_manager() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_6(mht_6_v, 265, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "resource_manager");

    return resource_manager_;
  }

  const gtl::InlinedVector<AllocatorAttributes, 4>& input_alloc_attrs() const {
    return input_alloc_attrs_;
  }
  const gtl::InlinedVector<AllocatorAttributes, 1>& output_alloc_attrs() const {
    return output_alloc_attrs_;
  }

 private:
  explicit OpKernelRunner(
      tensorflow::Device* device,
      tensorflow::FunctionLibraryRuntime* function_library_runtime,
      std::unique_ptr<OpKernel> op_kernel);

  tensorflow::Device* device_ = nullptr;
  tensorflow::FunctionLibraryRuntime* function_library_runtime_ = nullptr;
  tensorflow::ResourceMgr* resource_manager_ = nullptr;
  std::unique_ptr<OpKernel> op_kernel_;
  bool is_async_ = false;
  gtl::InlinedVector<AllocatorAttributes, 4> input_alloc_attrs_;
  gtl::InlinedVector<AllocatorAttributes, 1> output_alloc_attrs_;
};

// OpKernelRunState keeps the states needed for per-kernel execution.
struct OpKernelRunState {
  gtl::InlinedVector<tensorflow::Tensor, 4> input_tf_tensors;
  gtl::InlinedVector<tensorflow::TensorValue, 4> input_tf_tensor_values;
  OpKernelContext::Params params;

  OpKernelRunState() = default;
  OpKernelRunState(
      const gtl::InlinedVector<tensorflow::TensorValue, 4>& tensor_values,
      const OpKernelContext::Params& p) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_7(mht_7_v, 303, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "OpKernelRunState");

    // `input_tf_tensor_values` contains the reference to all tensor used,
    // while `input_tf_tensors` only contains those needs ownership so their
    // sizes may not match. For this copy assignment, we conservatively copy all
    // tensors.
    input_tf_tensors.reserve(tensor_values.size());
    for (const auto& tensor_value : tensor_values) {
      input_tf_tensors.push_back(*tensor_value.tensor);
    }
    for (auto& tensor : input_tf_tensors) {
      input_tf_tensor_values.emplace_back(&tensor);
    }

    // Since `input_tf_tensor_values` and `params` contains pointers to
    // `input_tf_tensors`, we need to change those pointers to the correct ones
    // after copying.
    params = p;
    params.inputs = &input_tf_tensor_values;
    // Clear eigen_gpu_device to ensure OpKernelContext constructor will make a
    // new eigen GPU device.
    params.eigen_gpu_device = nullptr;
  }

  OpKernelRunState(const OpKernelRunState& other) = delete;
  OpKernelRunState& operator=(const OpKernelRunState& other) = delete;

  ~OpKernelRunState() = default;
};

// OpKernelRunnerTable for keeping OpKernelRunner instances to avoid expensive
// reinstantiation of OpKernel and other repeated setup per kernel execution.
// OpKernelRunnerTable is thread-compatible.
class OpKernelRunnerTable {
 public:
  OpKernelRunnerTable() = default;

  // Return true if it successfully inserts `runner`. `index` is supposed to be
  // dense.
  bool Insert(int64_t index, OpKernelRunner runner) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_8(mht_8_v, 344, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "Insert");

    if (runners_.size() <= index) runners_.resize(index + 1);
    if (runners_[index].has_value()) return false;
    runners_[index] = std::move(runner);
    return true;
  }

  // Return the OpKernelRunner at the corresponding `index` in the table. The
  // result can never be nullptr. It is a fatal error to use an index that is
  // not in the table. Note that the returned pointer will be invalidated if
  // Insert() is called.
  const OpKernelRunner* Get(int64_t index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStfrtPSfallbackPSop_kernel_runnerDTh mht_9(mht_9_v, 358, "", "./tensorflow/core/tfrt/fallback/op_kernel_runner.h", "Get");

    // Out of bounds vector access will throw an exception and anyway will crash
    // the binary, prefer a more readable error message.
    CHECK_GT(runners_.size(), index)  // Crash OK
        << "runner index is out of bounds: index=" << index
        << " size=" << runners_.size();
    auto& result = runners_.at(index);
    CHECK(result.has_value())  // Crash OK
        << "runner is not available: index=" << index;
    return &(*result);
  }

 private:
  std::vector<absl::optional<OpKernelRunner>> runners_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
