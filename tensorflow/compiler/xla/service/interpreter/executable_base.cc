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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutable_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutable_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutable_baseDTcc() {
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

#include "tensorflow/compiler/xla/service/interpreter/executable_base.h"

#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace interpreter {

InterpreterExecutableBase::InterpreterExecutableBase(
    std::unique_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutable_baseDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/xla/service/interpreter/executable_base.cc", "InterpreterExecutableBase::InterpreterExecutableBase");
}

StatusOr<ExecutionOutput> InterpreterExecutableBase::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSexecutable_baseDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/xla/service/interpreter/executable_base.cc", "InterpreterExecutableBase::ExecuteAsyncOnStream");

  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();

  // Convert the ShapeTree to a ShapedBuffer. We do this so we can call
  // TransferManager methods below.
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  int device_ordinal = run_options->device_ordinal();
  if (device_ordinal < 0) {
    device_ordinal = 0;
  }
  for (auto& argument : arguments) {
    const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
    argument_buffers.push_back(ShapedBuffer(buffers.shape(),
                                            /*device_ordinal=*/device_ordinal));
    auto in_it = buffers.begin();
    auto out_it = argument_buffers.back().buffers().begin();
    for (; in_it != buffers.end(); ++in_it, ++out_it) {
      out_it->second = in_it->second.AsDeviceMemoryBase();
    }
  }

  VLOG(1) << "Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }

  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  const HloComputation* computation = module().entry_computation();
  if (computation->num_parameters() != arguments.size()) {
    return tensorflow::errors::Internal(
        "Mismatch between argument count and graph parameter count.");
  }

  // Check that the args have the right shape.
  for (int64_t i = 0; i < computation->num_parameters(); ++i) {
    const auto& expected_shape = computation->parameter_instruction(i)->shape();
    const auto& actual_shape = argument_buffers[i].on_device_shape();
    bool shape_match = true;
    if (expected_shape.is_dynamic()) {
      if (!ShapeUtil::DynamicArrayShapeIsCompatible(actual_shape,
                                                    expected_shape)) {
        shape_match = false;
      }
    } else if (!Shape::Equal().MinorToMajorOnlyInLayout()(expected_shape,
                                                          actual_shape)) {
      shape_match = false;
    }
    if (!shape_match) {
      return InvalidArgument(
          "Shape mismatch on parameter %d.  Expected %s, but was %s.", i,
          ShapeUtil::HumanStringWithLayout(expected_shape),
          ShapeUtil::HumanStringWithLayout(actual_shape));
    }
  }

  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                      TransferManager::GetForPlatform(platform));

  // Transform the ShapedBuffer arguments into literals which the evaluator
  // consumes.
  std::vector<Literal> arg_literals;
  const int64_t num_parameters = computation->num_parameters();
  arg_literals.reserve(num_parameters);
  for (int64_t p = 0; p < num_parameters; ++p) {
    TF_ASSIGN_OR_RETURN(Literal arg_literal,
                        transfer_manager->TransferLiteralFromDevice(
                            run_options->stream(), argument_buffers[p]));
    const auto& expected_shape = computation->parameter_instruction(p)->shape();
    if (expected_shape.is_dynamic()) {
      // Expand the input literal to expected shape.
      arg_literal = arg_literal.ToBoundedDynamic(expected_shape);
    }
    arg_literals.push_back(std::move(arg_literal));
  }

  TF_ASSIGN_OR_RETURN(Literal result_literal,
                      Evaluate(run_options, *computation, arg_literals));
  // Shrink the generated dynamic shape into static shape.
  result_literal = result_literal.ToStatic();

  // Transform the result literal back into a ShapedBuffer.
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer result_buffers,
                      transfer_manager->AllocateScopedShapedBuffer(
                          result_literal.shape(), run_options->allocator(),
                          executor->device_ordinal()));
  TF_RETURN_IF_ERROR(transfer_manager->TransferLiteralToDevice(
      run_options->stream(), result_literal, result_buffers));
  ExecutionOutput result(std::move(result_buffers));

  uint64_t end_micros = tensorflow::Env::Default()->NowMicros();

  ExecutionProfile* profile = run_options->run_options().execution_profile();
  if (profile) {
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    profile->set_compute_time_ns(std::max(nanoseconds, 1.0));
  }
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

}  // namespace interpreter
}  // namespace xla
