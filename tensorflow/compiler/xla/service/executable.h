/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh() {
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
#include <set>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

// TODO(b/150633678): Both the ExecutionInput and ExecutionOutput need to be
// revisited, with the execute APIs taking data structure which can better model
// shareable buffers.
//
// ExecutionInput buffers are in one of three states:
//
// 1) Owned by the caller and immutable.
// 2) Donated by the caller but returned on error.
// 3) Donated by the caller and freed on error.
//
// Case (1) buffers are stored as MaybeOwningDeviceMemory(DeviceMemoryBase).
// Case (2) buffers are stored as MaybeOwningDeviceMemory(OwningDeviceMemory),
//   with their indices present in unowned_indices_.
// Case (3) buffers are stored as MaybeOwningDeviceMemory(OwningDeviceMemory),
//   with their indices absent from unowned_indices_.
class ExecutionInput {
 public:
  explicit ExecutionInput(xla::Shape shape) : buffers_(std::move(shape)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_0(mht_0_v, 230, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionInput");

    SetHostShape(ShapeUtil::DeviceShapeToHostShape(buffers_.shape()));
  }
  // TODO(b/170310047): remove this overload.
  ExecutionInput(xla::Shape shape, xla::Shape host_shape)
      : buffers_(std::move(shape)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_1(mht_1_v, 238, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionInput");

    SetHostShape(ShapeUtil::DeviceShapeToHostShape(buffers_.shape()));
  }

  explicit ExecutionInput(ShapeTree<MaybeOwningDeviceMemory> buffers)
      : buffers_(std::move(buffers)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_2(mht_2_v, 246, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionInput");

    SetHostShape(ShapeUtil::DeviceShapeToHostShape(buffers_.shape()));
  }
  // TODO(b/170310047): remove this overload.
  ExecutionInput(ShapeTree<MaybeOwningDeviceMemory> buffers,
                 xla::Shape host_shape)
      : buffers_(std::move(buffers)) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_3(mht_3_v, 255, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionInput");

    SetHostShape(ShapeUtil::DeviceShapeToHostShape(buffers_.shape()));
  }

  ExecutionInput(ExecutionInput&&) = default;

  ~ExecutionInput();

  ExecutionInput& operator=(ExecutionInput&&) = default;

  const Shape& shape() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_4(mht_4_v, 268, "", "./tensorflow/compiler/xla/service/executable.h", "shape");

    return dynamic_shape_ != nullptr ? *dynamic_shape_ : buffers_.shape();
  }

  const Shape& host_shape() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_5(mht_5_v, 275, "", "./tensorflow/compiler/xla/service/executable.h", "host_shape");

    return host_shape_ != nullptr ? *host_shape_ : shape();
  }

  Status SetDynamicShape(Shape dynamic_shape);

  xla::StatusOr<xla::ShapedBuffer> ToShapedBuffer(
      se::DeviceMemoryAllocator* allocator, int device_ordinal) const;

  void SetBuffer(const ShapeIndex& index, MaybeOwningDeviceMemory buffer) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_6(mht_6_v, 287, "", "./tensorflow/compiler/xla/service/executable.h", "SetBuffer");

    *buffers_.mutable_element(index) = std::move(buffer);
  }

  void SetUnownedBuffer(const ShapeIndex& index,
                        MaybeOwningDeviceMemory buffer);

  void SetUnownedIndex(const ShapeIndex& index) {
    std::cout << "ExecutionInput::SetUnownedIndex with 1 argument" << std::endl;
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_7(mht_7_v, 297, "", "./tensorflow/compiler/xla/service/executable.h", "SetUnownedIndex");

    unowned_indices_.insert(index);
  }

  void ClearUnownedIndex(const ShapeIndex& index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_8(mht_8_v, 304, "", "./tensorflow/compiler/xla/service/executable.h", "ClearUnownedIndex");

    unowned_indices_.erase(index);
  }

  const std::set<ShapeIndex>& unowned_indices() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_9(mht_9_v, 311, "", "./tensorflow/compiler/xla/service/executable.h", "unowned_indices");
 return unowned_indices_; }

  const ShapeTree<MaybeOwningDeviceMemory>& Buffers() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_10(mht_10_v, 316, "", "./tensorflow/compiler/xla/service/executable.h", "Buffers");
 return buffers_; }

  ShapeTree<MaybeOwningDeviceMemory>* MutableBuffers() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_11(mht_11_v, 321, "", "./tensorflow/compiler/xla/service/executable.h", "MutableBuffers");
 return &buffers_; }

  MaybeOwningDeviceMemory* MutableBuffer(const ShapeIndex& index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_12(mht_12_v, 326, "", "./tensorflow/compiler/xla/service/executable.h", "MutableBuffer");

    return buffers_.mutable_element(index);
  }

  const MaybeOwningDeviceMemory& Buffer(const ShapeIndex& index) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_13(mht_13_v, 333, "", "./tensorflow/compiler/xla/service/executable.h", "Buffer");

    return buffers_.element(index);
  }

 private:
  void SetHostShape(xla::Shape host_shape) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_14(mht_14_v, 341, "", "./tensorflow/compiler/xla/service/executable.h", "SetHostShape");

    if (shape() != host_shape) {
      host_shape_ = absl::make_unique<Shape>(std::move(host_shape));
    }
  }

  ShapeTree<MaybeOwningDeviceMemory> buffers_;
  // Set of indices of buffers that should be returned to the caller if an error
  // occurs when enqueuing the computation.
  std::set<ShapeIndex> unowned_indices_;
  std::unique_ptr<Shape> dynamic_shape_;
  std::unique_ptr<Shape> host_shape_;
};

// ExecutionOutput encapsulates the output buffers of a execution and the
// leftover buffers to be released by the caller.
class ExecutionOutput {
 public:
  explicit ExecutionOutput(ScopedShapedBuffer result)
      : result_(std::move(result)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_15(mht_15_v, 363, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionOutput");
}
  ExecutionOutput(ScopedShapedBuffer result,
                  std::vector<se::OwningDeviceMemory> to_be_released)
      : result_(std::move(result)),
        to_be_released_(std::move(to_be_released)) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_16(mht_16_v, 370, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionOutput");
}
  // TODO(b/170310047): remove this overload.
  ExecutionOutput(Shape on_host_shape, Shape on_device_shape,
                  se::DeviceMemoryAllocator* allocator, int device_ordinal)
      : result_(std::move(on_device_shape), allocator, device_ordinal) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_17(mht_17_v, 377, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionOutput");
}
  ExecutionOutput(Shape on_device_shape, se::DeviceMemoryAllocator* allocator,
                  int device_ordinal)
      : result_(std::move(on_device_shape), allocator, device_ordinal) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_18(mht_18_v, 383, "", "./tensorflow/compiler/xla/service/executable.h", "ExecutionOutput");
}
  ExecutionOutput(ExecutionOutput&&) = default;
  ExecutionOutput& operator=(ExecutionOutput&&) = default;

  ~ExecutionOutput() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_19(mht_19_v, 390, "", "./tensorflow/compiler/xla/service/executable.h", "~ExecutionOutput");

    // If the ExecutionOutput has not been committed, and if there are aliased
    // indices, clear them off the ScopedShapedBuffer to prevent them to be
    // released.
    for (auto& index : aliased_indices_) {
      result_.set_buffer(se::OwningDeviceMemory(), index);
    }
  }

  void AddAliasedIndex(ShapeIndex index) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_20(mht_20_v, 402, "", "./tensorflow/compiler/xla/service/executable.h", "AddAliasedIndex");

    aliased_indices_.push_back(std::move(index));
  }

  void AddToBeReleased(se::OwningDeviceMemory mem) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_21(mht_21_v, 409, "", "./tensorflow/compiler/xla/service/executable.h", "AddToBeReleased");

    to_be_released_.push_back(std::move(mem));
  }

  // Should be called once it is known that the execute operation succeeded,
  // before returning the ExecutionOutput to the caller.
  ExecutionOutput& Commit() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_22(mht_22_v, 418, "", "./tensorflow/compiler/xla/service/executable.h", "Commit");

    aliased_indices_.clear();
    return *this;
  }

  const ScopedShapedBuffer& Result() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_23(mht_23_v, 426, "", "./tensorflow/compiler/xla/service/executable.h", "Result");
 return result_; }

  ScopedShapedBuffer* MutableResult() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_24(mht_24_v, 431, "", "./tensorflow/compiler/xla/service/executable.h", "MutableResult");
 return &result_; }

  ScopedShapedBuffer ConsumeResult() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_25(mht_25_v, 436, "", "./tensorflow/compiler/xla/service/executable.h", "ConsumeResult");

    aliased_indices_.clear();
    return std::move(result_);
  }

  const std::vector<se::OwningDeviceMemory>& ToBeReleased() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_26(mht_26_v, 444, "", "./tensorflow/compiler/xla/service/executable.h", "ToBeReleased");

    return to_be_released_;
  }

  std::vector<se::OwningDeviceMemory> ConsumeToBeReleased() {
    return std::move(to_be_released_);
  }

  std::vector<ShapeIndex> ConsumeAliasedIndices() {
    auto aliased = std::move(aliased_indices_);
    aliased_indices_.clear();
    return aliased;
  }

 private:
  ScopedShapedBuffer result_;

  // Leftover buffers for the caller to release. Elements in this list are
  // donated input memory buffers that are not reused by XLA as outputs.
  std::vector<se::OwningDeviceMemory> to_be_released_;

  // These are the indices in result_ which have been aliased from the caller.
  // If the execution operation fails, the caller should maintain ownership of
  // the buffer, so we track the indices here, and unless the ExecutionOutput is
  // committed, we remove them from the result_ before destruction.
  std::vector<ShapeIndex> aliased_indices_;

  // A shape table is a continuous region in memory that is used to hold the
  // runtime dimension sizes of dynamic output shapes.
  se::OwningDeviceMemory output_shape_table_;
};

// A given platform's compiler will produce an Executable -- this is a uniform
// interface that is used for launching compiled programs across platforms.
class Executable {
 public:
  explicit Executable(std::shared_ptr<HloModule> hlo_module)
      : hlo_module_(std::move(hlo_module)) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_27(mht_27_v, 484, "", "./tensorflow/compiler/xla/service/executable.h", "Executable");
}

  // TODO(b/172012028): Remove this constructor.
  explicit Executable(
      std::shared_ptr<HloModule> hlo_module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
      std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
      : hlo_module_(std::move(hlo_module)),
        hlo_profile_printer_data_(std::move(hlo_profile_printer_data)),
        hlo_profile_index_map_(std::move(hlo_profile_index_map)) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_28(mht_28_v, 496, "", "./tensorflow/compiler/xla/service/executable.h", "Executable");

    CHECK_EQ(hlo_profile_printer_data_.get() == nullptr,
             hlo_profile_index_map_.get() == nullptr);
  }
  virtual ~Executable() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_29(mht_29_v, 503, "", "./tensorflow/compiler/xla/service/executable.h", "~Executable");
}

  // Enqueues the compilation result on the provided stream, passing the given
  // arguments. This call is blocking and returns after the execution is done.
  //
  // If the hlo_execution_profile is provided as non-nullptr, profiling will be
  // enabled.
  //
  // Returns a shaped buffer containing the result of the computation.
  StatusOr<ScopedShapedBuffer> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile);

  // Starts the given program executing on the given stream/executor.
  //
  // `arguments` are ShapeTree containing the input parameters. For each element
  // in the shape tree, if the element holds the ownership of the memory, it is
  // considered donated and XLA will potentially reuse it as output buffers. For
  // all donated inputs, XLA is also responsible for freeing them.
  //
  // If an input is donated to XLA but is not reused as output, it is returned
  // as an leftover buffer for the caller to release.
  //
  // This call should be non-blocking and may return as soon as all of the
  // operations are enqueued for launch on the stream. Note that some
  // implementations may in fact block or may block in some circumstances (e.g.,
  // when profiling); i.e., asynchronous is a "may" not a "must".
  //
  // If the hlo_execution_profile is provided as non-nullptr, profiling will be
  // enabled. Note that profiling is tricky to use correctly, as the profiling
  // objects (when they exist) must out-live the task.
  virtual StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile);

  // Same as ExecuteAsyncOnStream(), but blocks waiting for the computation to
  // complete.
  StatusOr<ExecutionOutput> ExecuteOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile);

  virtual StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) = 0;

  // Same as ExecuteOnStream(), but runs this executable on multiple
  // streams. arguments[i] contains the arguments to the execution on
  // run_options[i]->stream() and the returned value is at index i of the
  // returned vector.
  virtual StatusOr<std::vector<ScopedShapedBuffer>> ExecuteOnStreams(
      absl::Span<const ServiceExecutableRunOptions> run_options,
      absl::Span<const absl::Span<const ShapedBuffer* const>> arguments);

  // Populates `hlo_execution_profile` from `executor`. This is implicit in any
  // Execute* API call that takes a hlo_execution_profile argument, but must be
  // called explicitly for other (async, for example) variants after the stream
  // has completed.
  virtual Status PopulateExecutionProfile(
      ExecutionProfile* execution_profile,
      HloExecutionProfile* hlo_execution_profile, se::Stream* stream) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_30(mht_30_v, 569, "", "./tensorflow/compiler/xla/service/executable.h", "PopulateExecutionProfile");

    return Status::OK();
  }

  // Convenience wrapper for calling Executable::ExecuteOnStream. Sets up a
  // timer for the execution, sets up HLO profiling if enabled, and fills in the
  // given ExecutionProfile if non-null.
  StatusOr<ScopedShapedBuffer> ExecuteOnStreamWrapper(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments);

  StatusOr<ExecutionOutput> ExecuteOnStreamWrapper(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments);

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStreamWrapper(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments);

  StatusOr<ExecutionOutput> ExecuteAsyncOnStreamWrapper(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments);

  const HloProfilePrinterData& hlo_profile_printer_data() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_31(mht_31_v, 595, "", "./tensorflow/compiler/xla/service/executable.h", "hlo_profile_printer_data");

    CHECK(hlo_profiling_enabled());
    return *hlo_profile_printer_data_;
  }

  const HloProfileIndexMap& hlo_profile_index_map() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_32(mht_32_v, 603, "", "./tensorflow/compiler/xla/service/executable.h", "hlo_profile_index_map");

    CHECK(hlo_profiling_enabled());
    return *hlo_profile_index_map_;
  }

  // Returns whether this executable was compiled with HLO profilings support
  // enabled. If not, the caller should not expect an hlo_execution_profile
  // passed to ExecuteOnStream above to be populated during execution.
  bool hlo_profiling_enabled() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_33(mht_33_v, 614, "", "./tensorflow/compiler/xla/service/executable.h", "hlo_profiling_enabled");

    return hlo_profile_printer_data_ != nullptr;
  }

  HloModule& module() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_34(mht_34_v, 621, "", "./tensorflow/compiler/xla/service/executable.h", "module");
 return *hlo_module_; }
  std::shared_ptr<HloModule> shared_module() const { return hlo_module_; }

  const bool has_module() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_35(mht_35_v, 627, "", "./tensorflow/compiler/xla/service/executable.h", "has_module");
 return hlo_module_ != nullptr; }

  const HloModuleConfig& module_config() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_36(mht_36_v, 632, "", "./tensorflow/compiler/xla/service/executable.h", "module_config");
 return hlo_module_->config(); }

  // The shape (including layout) that results from this execution. This is the
  // shape of the DeviceMemoryBase result value in ExecuteOnStream above.
  const Shape& result_shape() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_37(mht_37_v, 639, "", "./tensorflow/compiler/xla/service/executable.h", "result_shape");

    return hlo_module_->config().entry_computation_layout().result_shape();
  }

  // Returns the size of the executable in bytes. Returns -1 if this query is
  // not supported by the executable.
  //
  // Does not include the size of used libraries (e.g. cuDNN, Eigen, etc.).
  virtual int64_t SizeOfGeneratedCodeInBytes() const;

  // Dumping helpers.
  void set_hlo_proto(std::unique_ptr<xla::HloProto> hlo_proto) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_38(mht_38_v, 653, "", "./tensorflow/compiler/xla/service/executable.h", "set_hlo_proto");

    hlo_proto_ = std::move(hlo_proto);
  }
  bool dumping_snapshot() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_39(mht_39_v, 659, "", "./tensorflow/compiler/xla/service/executable.h", "dumping_snapshot");
 return hlo_proto_ != nullptr; }
  HloProto const* hlo_proto() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_40(mht_40_v, 663, "", "./tensorflow/compiler/xla/service/executable.h", "hlo_proto");
 return hlo_proto_.get(); }

  std::string& debug_info() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_41(mht_41_v, 668, "", "./tensorflow/compiler/xla/service/executable.h", "debug_info");
 return debug_info_; }
  void set_debug_info(const std::string& debug_info) {
   std::vector<std::string> mht_42_v;
   mht_42_v.push_back("debug_info: \"" + debug_info + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSexecutableDTh mht_42(mht_42_v, 673, "", "./tensorflow/compiler/xla/service/executable.h", "set_debug_info");

    debug_info_ = debug_info;
  }
  // Gather unused but donated buffers, return them to the caller of this API.
  // We don't free buffers inside this function since the caller could have
  // different preferences for buffer deallocation. For example, in TensorFlow,
  // buffers are mostly efficiently deallocated as soon as a program has been
  // launched. However, in XRT, the buffers are expected to be deallocated after
  // the program has finished since XRT doesn't support async deallocation.
  void MarkToBeReleasedArguments(absl::Span<ExecutionInput> arguments,
                                 ExecutionOutput& result);

 protected:
  // HloModule this was compiled from. BufferAssignment keeps pointers to
  // HloInstructions owned by the HloModule so we need to keep the HloModule
  // around.
  const std::shared_ptr<HloModule> hlo_module_;

  // The serialized HLO proto. Non-null only if dumping snapshots is enabled.
  std::unique_ptr<HloProto const> hlo_proto_;

  // Execution count, used to generate a unique filename for each dumped
  // execution.
  int64_t execution_count_ = 0;

  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data_;
  std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map_;

  // Generic debug information as a string.
  std::string debug_info_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_EXECUTABLE_H_
