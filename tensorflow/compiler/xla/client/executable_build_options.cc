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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc() {
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

#include "tensorflow/compiler/xla/client/executable_build_options.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

ExecutableBuildOptions& ExecutableBuildOptions::set_device_allocator(
    se::DeviceMemoryAllocator* allocator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_device_allocator");

  device_allocator_ = allocator;
  return *this;
}

se::DeviceMemoryAllocator* ExecutableBuildOptions::device_allocator() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_1(mht_1_v, 203, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::device_allocator");

  return device_allocator_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_ordinal(
    int device_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_2(mht_2_v, 211, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_device_ordinal");

  CHECK_GE(device_ordinal, 0);
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableBuildOptions::device_ordinal() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_3(mht_3_v, 220, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::device_ordinal");
 return device_ordinal_; }

DebugOptions* ExecutableBuildOptions::mutable_debug_options() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_4(mht_4_v, 225, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::mutable_debug_options");

  if (!has_debug_options()) {
    debug_options_ = GetDebugOptionsFromFlags();
  }
  return &debug_options_.value();
}

ExecutableBuildOptions& ExecutableBuildOptions::set_result_layout(
    const Shape& shape_with_layout) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_5(mht_5_v, 236, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_result_layout");

  result_layout_set_ = true;
  result_layout_ = shape_with_layout;
  return *this;
}

const Shape* ExecutableBuildOptions::result_layout() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_6(mht_6_v, 245, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::result_layout");

  return result_layout_set_ ? &result_layout_ : nullptr;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_replicas(
    int num_replicas) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_7(mht_7_v, 253, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_num_replicas");

  num_replicas_ = num_replicas;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_partitions(
    int num_partitions) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_8(mht_8_v, 262, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_num_partitions");

  num_partitions_ = num_partitions;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_use_spmd_partitioning(
    bool use_spmd_partitioning) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_9(mht_9_v, 271, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_use_spmd_partitioning");

  use_spmd_partitioning_ = use_spmd_partitioning;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_use_auto_spmd_partitioning(
    bool use_auto_spmd_partitioning) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_10(mht_10_v, 280, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_use_auto_spmd_partitioning");

  use_auto_spmd_partitioning_ = use_auto_spmd_partitioning;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_deduplicate_hlo(
    bool deduplicate_hlo) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_11(mht_11_v, 289, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_deduplicate_hlo");

  deduplicate_hlo_ = deduplicate_hlo;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_assignment(
    const DeviceAssignment& device_assignment) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_12(mht_12_v, 298, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::set_device_assignment");

  device_assignment_ = device_assignment;
  return *this;
}

std::string ExecutableBuildOptions::ToString() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_13(mht_13_v, 306, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "ExecutableBuildOptions::ToString");

  std::string result_layout = "nullopt";
  if (result_layout_set_) {
    result_layout = ShapeUtil::HumanStringWithLayout(result_layout_);
  }
  return absl::StrFormat(
      "ExecutableBuildOptions{device_ordinal=%d, result_layout=%s, "
      "num_replicas=%d}",
      device_ordinal_, result_layout, num_replicas_);
}

ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTcc mht_14(mht_14_v, 322, "", "./tensorflow/compiler/xla/client/executable_build_options.cc", "CreateExecutionOptions");

  ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  if (build_options.has_debug_options()) {
    *execution_options.mutable_debug_options() = build_options.debug_options();
  }
  if (build_options.result_layout() != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        build_options.result_layout()->ToProto();
  } else {
    Shape result_shape(program_shape->result());
    LayoutUtil::SetToDefaultLayout(&result_shape);
    *execution_options.mutable_shape_with_output_layout() =
        result_shape.ToProto();
  }
  execution_options.set_num_replicas(build_options.num_replicas());
  execution_options.set_num_partitions(build_options.num_partitions());
  execution_options.set_use_spmd_partitioning(
      build_options.use_spmd_partitioning());
  execution_options.set_use_auto_spmd_partitioning(
      build_options.use_auto_spmd_partitioning());
  execution_options.set_deduplicate_hlo(build_options.deduplicate_hlo());
  execution_options.set_allow_spmd_sharding_propagation_to_output(
      build_options.allow_spmd_sharding_propagation_to_output());
  if (build_options.has_device_assignment()) {
    TF_CHECK_OK(build_options.device_assignment().Serialize(
        execution_options.mutable_device_assignment()));
  }
  execution_options.set_alias_passthrough_params(
      build_options.alias_passthrough_params());
  return execution_options;
}

}  // namespace xla
