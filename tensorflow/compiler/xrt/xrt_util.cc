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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc() {
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

#include "tensorflow/compiler/xrt/xrt_util.h"

#include <stdlib.h>
#include <string.h>

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {

mutex nccl_factory_mutex(LINKER_INITIALIZED);
std::shared_ptr<NcclUniqueIdFactory>* nccl_factory;

// The ScopedHandles data structure is used in the ExecuteChained() API and its
// task is to track tuple allocation registrations. It is used both the track
// intermediate results of a chained computation, or its final results. Anything
// which is marked to be released, will be released using the XRTMemoryManager
// once the object is destroyed (unless an explicit call to Drop() or Release()
// is made).
class ScopedHandles {
 public:
  explicit ScopedHandles(RefPtr<XRTMemoryManager> memory_manager)
      : memory_manager_(std::move(memory_manager)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_0(mht_0_v, 210, "", "./tensorflow/compiler/xrt/xrt_util.cc", "ScopedHandles");
}

  ~ScopedHandles() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xrt/xrt_util.cc", "~ScopedHandles");

    for (size_t i = 0; i < handles_.size(); ++i) {
      if (handles_release_[i]) {
        memory_manager_->Release(handles_[i]).IgnoreError();
      }
    }
  }

  int64_t operator[](size_t index) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xrt/xrt_util.cc", "lambda");
 return handles_.at(index); }

  size_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/xrt/xrt_util.cc", "size");
 return handles_.size(); }

  // Adds the given handle at the index position, by marking it releasable
  // according to the release argument. If an existing, and to-be-released
  // handle already exists at the same index, it will be released.
  Status Add(size_t index, int64_t handle, bool release) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_4(mht_4_v, 239, "", "./tensorflow/compiler/xrt/xrt_util.cc", "Add");

    if (index >= handles_.size()) {
      handles_.resize(index + 1, XRTMemoryManager::InvalidKey());
      handles_release_.resize(index + 1, false);
    }
    if (handles_release_[index]) {
      Status status = memory_manager_->Release(handles_[index]);
      if (!status.ok()) {
        if (release) {
          memory_manager_->Release(handle).IgnoreError();
        }
        return status;
      }
    }
    handles_[index] = handle;
    handles_release_[index] = release;
    return Status::OK();
  }

  // Adds a to-be-released tuple allocation at the given index.
  Status Add(size_t index, RefPtr<XRTTupleAllocation> tuple) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xrt/xrt_util.cc", "Add");

    return Add(index, memory_manager_->Register(std::move(tuple)),
               /*release=*/true);
  }

  // Drops the handle at the given index, and releases it using the
  // XRTMemoryManager::Release() if marked as to-be-released.
  Status Drop(size_t index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_6(mht_6_v, 272, "", "./tensorflow/compiler/xrt/xrt_util.cc", "Drop");

    if (handles_release_.at(index)) {
      TF_RETURN_IF_ERROR(memory_manager_->Release(handles_[index]));
    }
    Release(index);
    return Status::OK();
  }

  // Releases the handle at the given index. The destructor will not use that
  // XRTMemoryManager::Release() API on such handle.
  int64_t Release(size_t index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_7(mht_7_v, 285, "", "./tensorflow/compiler/xrt/xrt_util.cc", "Release");

    int64_t handle = handles_.at(index);
    handles_[index] = XRTMemoryManager::InvalidKey();
    handles_release_[index] = false;
    return handle;
  }

  // Looks up the handle stored at the given index, and returns the matching
  // tuple allocation.
  xla::StatusOr<RefPtr<XRTTupleAllocation>> Lookup(size_t index) const {
    return memory_manager_->Lookup(handles_.at(index));
  }

 private:
  RefPtr<XRTMemoryManager> memory_manager_;
  std::vector<int64_t> handles_;
  std::vector<bool> handles_release_;
};

bool DebugOptionsPassThroughEnabled() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_8(mht_8_v, 307, "", "./tensorflow/compiler/xrt/xrt_util.cc", "DebugOptionsPassThroughEnabled");

  const char* env = getenv("TF_XLA_DEBUG_OPTIONS_PASSTHROUGH");
  bool enabled =
      env != nullptr && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
  if (enabled) {
    LOG(WARNING) << "Passing through XLA debug options!";
  } else {
    LOG(WARNING) << "TF_XLA_DEBUG_OPTIONS_PASSTHROUGH not set, not all options "
                    "will be retained";
  }
  return enabled;
}

string SafeDebugPath(const string& path) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_9(mht_9_v, 324, "", "./tensorflow/compiler/xrt/xrt_util.cc", "SafeDebugPath");

  if (path.empty() || path.compare(0, 5, "gs://") == 0 ||
      path.compare(0, 11, "bigstore://") == 0) {
    return path;
  }
  LOG(WARNING) << "Invalid config path (will be dropped): " << path;
  return string();
}

Status MakeOutput(const RefPtr<XRTTupleAllocation>& output, int64_t index,
                  RefPtr<XRTTupleAllocation>* result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_10(mht_10_v, 337, "", "./tensorflow/compiler/xrt/xrt_util.cc", "MakeOutput");

  if (index == 0) {
    *result = output;
  } else {
    XRTTupleAllocation* tuple;
    TF_RETURN_IF_ERROR(
        XRTTupleAllocation::MakeSubBuffer(output.get(), {index - 1}, &tuple,
                                          /*alias_parent_allocation=*/true));
    result->reset(tuple);
  }
  return Status::OK();
}

Status PopulateOpWorkingSet(xla::Backend* backend,
                            const xrt::XRTChainedExecuteOp& op,
                            int current_index, const ScopedHandles& outputs,
                            XRTMemoryManager::WorkingSet* working_set,
                            se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_11(mht_11_v, 357, "", "./tensorflow/compiler/xrt/xrt_util.cc", "PopulateOpWorkingSet");

  for (int i = 0; i < op.inputs_size(); ++i) {
    auto& input = op.inputs(i);
    if (input.op_index() >= current_index) {
      return errors::InvalidArgument(
          "Input index ", input.op_index(),
          " is above the current position: ", current_index);
    }
    TF_RETURN_IF_ERROR(working_set->LookupAndPin(
        backend, outputs[input.op_index()], allocator));
  }
  return Status::OK();
}

}  // namespace

void SetNcclUniqueIdFactory(std::shared_ptr<NcclUniqueIdFactory> factory) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_12(mht_12_v, 376, "", "./tensorflow/compiler/xrt/xrt_util.cc", "SetNcclUniqueIdFactory");

  mutex_lock lock(nccl_factory_mutex);
  if (nccl_factory == nullptr) {
    nccl_factory = new std::shared_ptr<NcclUniqueIdFactory>();
  }
  *nccl_factory = std::move(factory);
}

std::shared_ptr<NcclUniqueIdFactory> GetNcclUniqueIdFactory() {
  mutex_lock lock(nccl_factory_mutex);
  return nccl_factory != nullptr ? *nccl_factory : nullptr;
}

xla::DebugOptions BuildXlaDebugOptions(const xla::DebugOptions& ref_options) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_13(mht_13_v, 392, "", "./tensorflow/compiler/xrt/xrt_util.cc", "BuildXlaDebugOptions");

  static const bool options_passthrough = DebugOptionsPassThroughEnabled();
  if (options_passthrough) {
    return ref_options;
  }
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  options.set_xla_dump_to(SafeDebugPath(ref_options.xla_dump_to()));
  options.set_xla_dump_hlo_as_proto(ref_options.xla_dump_hlo_as_proto());
  options.set_xla_dump_hlo_as_text(ref_options.xla_dump_hlo_as_text());
  options.set_xla_dump_hlo_snapshots(ref_options.xla_dump_hlo_snapshots());
  options.set_xla_dump_hlo_pass_re(ref_options.xla_dump_hlo_pass_re());
  options.set_xla_dump_include_timestamp(
      ref_options.xla_dump_include_timestamp());
  options.set_xla_dump_max_hlo_modules(ref_options.xla_dump_max_hlo_modules());
  for (auto& pass : ref_options.xla_disable_hlo_passes()) {
    options.add_xla_disable_hlo_passes(pass);
  }
  return options;
}

xla::StatusOr<std::vector<InputCoords>> GetComputationInputs(
    OpKernelContext* context, const char* input_name) {
  OpInputList arg_list;
  TF_RETURN_IF_ERROR(context->input_list(input_name, &arg_list));
  // Concatenate all input uids from list of scalars-or-vectors carrying them.
  std::vector<InputCoords> input_coords;
  for (int i = 0; i < arg_list.size(); ++i) {
    const Tensor& arg = arg_list[i];
    if (TensorShapeUtils::IsScalar(arg.shape())) {
      input_coords.emplace_back(arg.scalar<int64_t>()());
    } else {
      TF_RET_CHECK(TensorShapeUtils::IsVector(arg.shape()));
      auto arg_vec = arg.vec<int64_t>();
      const int64_t num_elts = arg.shape().dim_size(0);
      for (int i = 0; i < num_elts; ++i) {
        input_coords.emplace_back(arg_vec(i));
      }
    }
  }
  return std::move(input_coords);
}

bool InputShapeMatches(const xla::Shape& parameter_shape,
                       const xla::Shape& input_shape) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_14(mht_14_v, 438, "", "./tensorflow/compiler/xrt/xrt_util.cc", "InputShapeMatches");

  auto shape_checker = [&](const xla::Shape& pshape,
                           const xla::ShapeIndex& index) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_15(mht_15_v, 443, "", "./tensorflow/compiler/xrt/xrt_util.cc", "lambda");

    if (pshape.IsArray()) {
      TF_ASSIGN_OR_RETURN(const xla::Shape* ishape,
                          xla::ShapeUtil::TryGetSubshape(input_shape, index));
      if (pshape.rank() != ishape->rank() ||
          pshape.element_type() != ishape->element_type()) {
        return errors::InvalidArgument("Mismatching shapes");
      }
      if (pshape.is_static() && pshape.layout() != ishape->layout()) {
        return errors::InvalidArgument("Mismatching layouts");
      }
      for (int64_t dim = 0; dim < pshape.rank(); ++dim) {
        if (pshape.is_dynamic_dimension(dim)) {
          if (pshape.dimensions(dim) < ishape->dimensions(dim)) {
            return errors::InvalidArgument("Mismatching shapes");
          }
        } else if (pshape.dimensions(dim) != ishape->dimensions(dim)) {
          return errors::InvalidArgument("Mismatching shapes");
        }
      }
    }
    return Status::OK();
  };
  return xla::ShapeUtil::ForEachSubshapeWithStatus(parameter_shape,
                                                   shape_checker)
      .ok();
}

xla::StatusOr<std::vector<RefPtr<XRTTupleAllocation>>> GetInputTupleAllocations(
    const std::vector<InputCoords>& input_coords,
    XRTMemoryManager::WorkingSet* working_set, xla::Backend* backend,
    int64_t num_input_shapes,
    const std::function<xla::Shape(int64_t)>& shape_getter, bool release_inputs,
    se::DeviceMemoryAllocator* allocator) {
  if (input_coords.size() != num_input_shapes) {
    return errors::InvalidArgument(
        "Number of inputs does not match executable proto input shapes: ",
        input_coords.size(), " vs. ", num_input_shapes);
  }
  std::vector<RefPtr<XRTTupleAllocation>> input_tuples;
  input_tuples.reserve(input_coords.size());
  for (size_t i = 0; i < input_coords.size(); ++i) {
    TF_RETURN_IF_ERROR(
        working_set->LookupAndPin(backend, input_coords[i].handle, allocator));
    auto tuple = working_set->PinnedTuples().back();
    if (release_inputs) {
      // We are holding a reference to the tuple, so we can safely delete it
      // from the resource manager here.
      TF_RETURN_IF_ERROR(
          working_set->MemoryManager()->Release(input_coords[i].handle));
      VLOG(2) << "Released allocation handle " << input_coords[i].handle;
    }
    xla::Shape input_shape = shape_getter(i);
    if (!InputShapeMatches(input_shape, tuple->on_host_shape())) {
      return errors::InvalidArgument(
          "Run-time shape mismatch for XRTExecute argument[", i, "] (",
          input_coords[i].handle, "). Expected ", input_shape.DebugString(),
          "; got ", tuple->on_host_shape().DebugString());
    }
    if (input_coords[i].index.empty()) {
      input_tuples.emplace_back(std::move(tuple));
    } else {
      XRTTupleAllocation* sub_tuple;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          tuple.get(), input_coords[i].index, &sub_tuple,
          /*alias_parent_allocation=*/true));
      input_tuples.emplace_back(sub_tuple);
    }
  }
  return std::move(input_tuples);
}

Status RebuildOutputAliases(
    const RefPtr<XRTTupleAllocation>& output_tuple,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const xla::HloInputOutputAliasConfig& input_output_alias) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_16(mht_16_v, 521, "", "./tensorflow/compiler/xrt/xrt_util.cc", "RebuildOutputAliases");

  auto alias_function =
      [&](const xla::ShapeIndex& output_index,
          const xla::HloInputOutputAliasConfig::Alias& alias) -> Status {
    TF_RET_CHECK(alias.parameter_number < input_tuples.size());
    return output_tuple->AliasBufferFrom(*input_tuples[alias.parameter_number],
                                         alias.parameter_index, output_index);
  };
  return input_output_alias.ForEachAliasWithStatus(alias_function);
}

xla::StatusOr<std::vector<xla::ExecutionInput>> GetArgumentsBuffers(
    const xla::HloInputOutputAliasConfig& input_output_alias,
    absl::Span<const RefPtr<XRTTupleAllocation>> input_tuples,
    const std::vector<bool>& input_is_dynamic, bool release_inputs) {
  auto is_dynamic = [&](size_t arg) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_17(mht_17_v, 539, "", "./tensorflow/compiler/xrt/xrt_util.cc", "lambda");

    return arg < input_is_dynamic.size() && input_is_dynamic[arg];
  };
  std::vector<xla::ExecutionInput> arguments;
  // Don't alias dynamic input -- Due to the underlying implementation,
  // aliased inputs have two owners: XRTAllocation and return value of
  // this function. If an argument is dynamic and the ownership is
  // released to output of this function, TPUExecute will free it and
  // reallocate a new one, which creates a double freeing issue where
  // XRTAllocation also attempts to release the buffer.
  bool alias_outputs = release_inputs && input_tuples.size() == 1 &&
                       input_tuples[0]->IsExclusiveOwner() && !is_dynamic(0);
  arguments.reserve(input_tuples.size());
  for (int64_t i = 0; i < input_tuples.size(); ++i) {
    auto alias_checker =
        [&](const xla::ShapeIndex& index) -> xla::StatusOr<bool> {
      if (input_output_alias.ParameterHasAlias(i, index)) {
        TF_RET_CHECK(!is_dynamic(i));
        return true;
      }
      return alias_outputs;
    };
    TF_ASSIGN_OR_RETURN(xla::ExecutionInput exec_input,
                        input_tuples[i]->ToExecutionInput(alias_checker));
    arguments.emplace_back(std::move(exec_input));
  }
  return std::move(arguments);
}

Status CreateExecuteOutput(OpKernelContext* context,
                           XRTMemoryManager* memory_manager,
                           RefPtr<XRTTupleAllocation> output_tuple,
                           bool return_exploded_tuple) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_18(mht_18_v, 574, "", "./tensorflow/compiler/xrt/xrt_util.cc", "CreateExecuteOutput");

  if (return_exploded_tuple && output_tuple->on_host_shape().IsTuple()) {
    int64_t tuple_element_count =
        xla::ShapeUtil::TupleElementCount(output_tuple->on_device_shape());
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({tuple_element_count}), &output_tensor));

    for (int64_t i = 0; i < tuple_element_count; ++i) {
      XRTTupleAllocation* suballocation;
      TF_RETURN_IF_ERROR(XRTTupleAllocation::MakeSubBuffer(
          output_tuple.get(), {i}, &suballocation,
          /*alias_parent_allocation=*/false));
      output_tensor->vec<int64_t>()(i) =
          memory_manager->Register(suballocation);
    }
  } else {
    Tensor* output_tensor;
    TF_RETURN_IF_ERROR(
        context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<int64_t>()() =
        memory_manager->Register(std::move(output_tuple));
  }
  return Status::OK();
}

Status ExecuteChained(OpKernelContext* context,
                      const RefPtr<XRTMemoryManager>& memory_manager,
                      xla::Backend* backend, int device_ordinal,
                      const xrt::XRTChainedExecutePlan& plan,
                      const xrt::XRTChainedExecuteConfig& config,
                      const ChainedExecuteFn& execute_op,
                      se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_utilDTcc mht_19(mht_19_v, 609, "", "./tensorflow/compiler/xrt/xrt_util.cc", "ExecuteChained");

  // Create the vector which tracks the uses of the intermediate chained
  // operations outputs.
  std::vector<int64_t> uses(plan.ops_size(), 0);
  for (auto& op : plan.ops()) {
    for (auto& input : op.inputs()) {
      uses[input.op_index()] += 1;
    }
  }

  ScopedHandles outputs(memory_manager);
  ScopedHandles results(memory_manager);
  for (int i = 0; i < plan.ops_size(); ++i) {
    auto& op = plan.ops(i);
    if (op.op_oneof_case() == xrt::XRTChainedExecuteOp::kDataHandle) {
      // This operation is a device data load. Set the handle as output and
      // leave the release flag off, since this is not an intermediate output.
      TF_RETURN_IF_ERROR(outputs.Add(i, op.data_handle(), /*release=*/false));
    } else if (op.op_oneof_case() ==
               xrt::XRTChainedExecuteOp::kComputationHandle) {
      // This is an XRT execute operation, forward to the device specific
      // handler. Populating the working set makes sure the input allocations
      // for this execute operations are pinned to device memory.
      XRTMemoryManager::WorkingSet working_set(memory_manager);
      TF_RETURN_IF_ERROR(PopulateOpWorkingSet(backend, op, i, outputs,
                                              &working_set, allocator));
      TF_ASSIGN_OR_RETURN(auto tuple,
                          execute_op(op, working_set.PinnedTuples()));
      TF_RETURN_IF_ERROR(outputs.Add(i, std::move(tuple)));
    } else {
      return errors::InvalidArgument(
          "Undefined operation kind at post-order position ", i);
    }
    // If the result of this chained operation is an output result, feed the
    // results at the desired position.
    for (auto& output : op.outputs()) {
      TF_ASSIGN_OR_RETURN(auto tuple, outputs.Lookup(i));
      RefPtr<XRTTupleAllocation> result;
      TF_RETURN_IF_ERROR(MakeOutput(tuple, output.output_index(), &result));
      TF_RETURN_IF_ERROR(results.Add(output.result_index(), std::move(result)));
    }
    // Drop intermediate results which have no more users.
    for (auto& input : op.inputs()) {
      uses[input.op_index()] -= 1;
      if (uses[input.op_index()] == 0) {
        TF_RETURN_IF_ERROR(outputs.Drop(input.op_index()));
      }
    }
  }

  Tensor* output_tensor;
  TF_RETURN_IF_ERROR(context->allocate_output(
      0, TensorShape({static_cast<int64_t>(results.size())}), &output_tensor));
  for (size_t i = 0; i < results.size(); ++i) {
    output_tensor->vec<int64_t>()(i) = results.Release(i);
  }
  return Status::OK();
}

}  // namespace tensorflow
