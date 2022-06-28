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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"

#include <cassert>

#include "tensorflow/compiler/xla/cpu_function_runtime.h"

namespace tensorflow {

XlaCompiledCpuFunction::XlaCompiledCpuFunction(const StaticData& static_data,
                                               AllocMode alloc_mode)
    : raw_function_(static_data.raw_function_),
      result_index_(static_data.result_index_),
      buffer_table_(new void*[static_data.num_buffers_]),
      buffer_infos_(static_data.buffer_infos_),
      arg_index_table_(static_data.arg_index_table_),
      num_args_(static_data.num_args_),
      num_variables_(static_data.num_variables_),
      arg_names_(static_data.arg_names_),
      variable_names_(static_data.variable_names_),
      result_names_(static_data.result_names_),
      program_shape_(static_data.program_shape_),
      hlo_profile_printer_data_(static_data.hlo_profile_printer_data_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::XlaCompiledCpuFunction");

  bool allocate_entry_params =
      alloc_mode == AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS;
  // Allocate arg and temp buffers.
  alloc_buffer_table_ = xla::cpu_function_runtime::MallocContiguousBuffers(
      static_data.buffer_infos_, static_data.num_buffers_,
      /*allocate_entry_params=*/allocate_entry_params, buffer_table_,
      /*annotate_initialized=*/true);
  // If Hlo profiling is enabled the generated code expects an appropriately
  // sized buffer to be passed in as the last argument.  If Hlo profiling is
  // disabled the last function argument is still present in the function
  // signature, but it is ignored by the generated code and we pass in null for
  // it.
  if (hlo_profiling_enabled()) {
    profile_counters_ = new int64_t[static_data.profile_counters_size_]();
  }
}

bool XlaCompiledCpuFunction::Run() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::Run");

  XlaCustomCallStatus status;
  raw_function_(buffer_table_[result_index_], &run_options_, nullptr,
                buffer_table_, &status, profile_counters_);
  return !xla::CustomCallStatusGetMessage(&status).has_value();
}

XlaCompiledCpuFunction::~XlaCompiledCpuFunction() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::~XlaCompiledCpuFunction");

  xla::cpu_function_runtime::FreeContiguous(alloc_buffer_table_);
  delete[] buffer_table_;
  delete[] profile_counters_;
}

namespace {

constexpr int kNotFound = -1;

// Linear search through `names` looking for a match with `name`. Returns -1 if
// the name isn't found, or is empty.
//
// REQUIRES: `names` is a nullptr-terminated array.
int LookupNameIndex(const string& name, const char** names) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_3(mht_3_v, 255, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "LookupNameIndex");

  // Hitting this assert means that there is no name-to-index data available;
  // for AOT try the setting the tfcompile --gen_name_to_index flag.
  assert(names != nullptr);

  if (name.empty()) {
    return kNotFound;
  }
  for (int index = 0; names[index] != nullptr; ++index) {
    if (name == names[index]) {
      return index;
    }
  }
  return kNotFound;
}

}  // namespace

int XlaCompiledCpuFunction::LookupArgIndex(const string& name) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::LookupArgIndex");

  return LookupNameIndex(name, arg_names_);
}

int XlaCompiledCpuFunction::LookupVariableIndex(const string& name) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_5(mht_5_v, 285, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::LookupVariableIndex");

  int index = LookupNameIndex(name, variable_names_);
  if (index == kNotFound) {
    return kNotFound;
  }
  return num_args_ - num_variables_ + index;
}

int XlaCompiledCpuFunction::LookupResultIndex(const string& name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTcc mht_6(mht_6_v, 297, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.cc", "XlaCompiledCpuFunction::LookupResultIndex");

  return LookupNameIndex(name, result_names_);
}

}  // namespace tensorflow
