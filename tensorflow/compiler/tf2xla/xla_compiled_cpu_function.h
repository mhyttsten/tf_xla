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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh() {
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


#include <cassert>
#include <string>

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/core/platform/types.h"

// Forward-declare, rather than include, to reduce code size for users that
// never use this functionality.
namespace xla {
class ProgramShapeProto;
class HloProfilePrinterData;
}  // namespace xla

namespace tensorflow {

// Represents a function compiled by XLA, produced via either JIT or AOT.
//
// The Run method invokes the actual computation, with inputs read from arg
// buffers, and outputs written to result buffers. Each Run call may also use a
// set of temporary buffers for the computation.
//
// By default each instance of this class manages its own arg, result and temp
// buffers. The AllocMode constructor parameter may be used to modify the buffer
// allocation strategy.
//
// Under the default allocation strategy, this class is thread-compatible:
// o Calls to non-const methods require exclusive access to the object.
// o Concurrent calls to const methods are OK, if those calls are made while it
//   is guaranteed that no thread may call a non-const method.
class XlaCompiledCpuFunction {
 public:
  // Type of the raw function, produced by either JIT or AOT.
  using RawFunction = void (*)(void* result,
                               const xla::ExecutableRunOptions* run_options,
                               const void** args, void** temps,
                               XlaCustomCallStatus*, int64_t* profile_counters);

  // StaticData represents the state necessary to run an XLA-compiled
  // function. For JIT this is backed by data in XlaJitCompiledCpuFunction; for
  // AOT this is backed by data compiled into the object file.
  //
  // The contents of StaticData are XLA-internal implementation details and
  // should not be relied on by clients (and therefore are private).
  class StaticData {
   private:
    // The raw function to call.
    RawFunction raw_function_;

    // Contains information about the buffers used by the XLA computation.
    const xla::cpu_function_runtime::BufferInfo* buffer_infos_ = nullptr;
    size_t num_buffers_ = 0;

    // Entry parameter i is described by
    // buffer_infos[arg_index_table[i]].
    const int32* arg_index_table_ = nullptr;

    // There are num_args entry parameters.
    int64_t num_args_ = 0;

    // There are num_variables variables.
    int64_t num_variables_ = 0;

    // The 0-based index of the result tuple, in the temp buffers.
    size_t result_index_ = 0;

    // [Optional] Arrays of arg and result names. These are arrays of C-style
    // strings, where the array is terminated by nullptr.
    const char** arg_names_ = nullptr;
    const char** variable_names_ = nullptr;
    const char** result_names_ = nullptr;

    // [Optional] Arg and result shapes.
    const xla::ProgramShapeProto* program_shape_ = nullptr;

    // [Optional] Profile printer data.  Null if profiling is disabled.
    const xla::HloProfilePrinterData* hlo_profile_printer_data_ = nullptr;

    // [Optional] The number of profile counters expected in the profile counter
    // buffer by the generated code and hlo_profile_printer.  0 if profiling is
    // disabled.  This information is already present in
    // hlo_profile_printer_data but xla::HloProfilePrinterData is forward
    // declared so we don't have access to that information here.
    int64_t profile_counters_size_ = 0;

    // Only XlaCompiledCpuFunction is allowed to read and write the above
    // fields.
    friend class XlaCompiledCpuFunction;
  };

  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    // Allocate all buffers - args, results, profile and temps.
    ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS,

    // Only allocate result, profile and temp buffers.
    // Use set_arg_data to set argument buffers before Run is called.
    RESULTS_PROFILES_AND_TEMPS_ONLY,
  };

  explicit XlaCompiledCpuFunction(
      const StaticData& static_data,
      AllocMode alloc_mode =
          AllocMode::ARGS_VARIABLES_RESULTS_PROFILES_AND_TEMPS);
  virtual ~XlaCompiledCpuFunction();

  XlaCompiledCpuFunction(const XlaCompiledCpuFunction&) = delete;
  XlaCompiledCpuFunction& operator=(const XlaCompiledCpuFunction&) = delete;

  // Sets the intra-op thread pool used to run individual ops concurrently.
  void set_thread_pool(const Eigen::ThreadPoolDevice* pool) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_0(mht_0_v, 299, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_thread_pool");

    run_options_.set_intra_op_thread_pool(pool);
  }

  // Runs the computation, with inputs read from arg buffers, and outputs
  // written to result buffers. Returns true on success and false on failure.
  bool Run();

  // Returns the error message from the previous failed Run call.
  //
  // TODO(fschneider): For now this always returns an empty string because there
  // is no support for error reporting in XLA. Remove this once all callers are
  // updated.
  string error_msg() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_1(mht_1_v, 315, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "error_msg");
 return {}; }

  // ------------------------------
  // Arg methods for managing input buffers. Buffers are in row-major order.

  // Returns the buffer for the positional argument at the given `index`.
  void* arg_data(size_t index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_2(mht_2_v, 324, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "arg_data");

    return buffer_table_[arg_index_table_[index]];
  }
  const void* arg_data(size_t index) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_3(mht_3_v, 330, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "arg_data");

    return buffer_table_[arg_index_table_[index]];
  }

  int num_args() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_4(mht_4_v, 337, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "num_args");
 return num_args_; }

  int num_variables() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_5(mht_5_v, 342, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "num_variables");
 return num_variables_; }

  // Returns the size of entry parameter `idx`.
  //
  // There is a static version of this method on tfcompile generated subclasses
  // of XlaCompiledCpuFunction, but try to prefer this when possible since it
  // works both for XlaJitCompiledCpuFunction and AOT compiled subclasses.
  int arg_size(int idx) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_6(mht_6_v, 352, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "arg_size");

    assert(idx < num_args());
    return buffer_infos_[arg_index_table_[idx]].size();
  }

  // Sets the buffer for the positional argument at the given `index` to `data`.
  // Must be called before Run to have an effect. May be called under any
  // AllocMode; if the AllocMode is RESULTS_AND_TEMPS_ONLY, this method must be
  // called for each positional argument, in order to set the argument buffers.
  //
  // Allocated memory must be aligned to the size specified by
  // xla::cpu_function_runtime::MinAlign(). If possible, use the functions in
  // tensorflow/compiler/tf2xla/cpu_function_runtime.h to ensure correct
  // alignment.
  //
  // Aliasing of argument and result buffers is not allowed, and results in
  // undefined behavior.
  void set_arg_data(size_t index, const void* data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_7(mht_7_v, 372, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_arg_data");

    assert((arg_size(index) < xla::cpu_function_runtime::MinAlign() ||
            (uintptr_t)data % xla::cpu_function_runtime::MinAlign() == 0) &&
           "Underaligned pointer!");
    // The const_cast is safe because the generated code does not write to arg
    // buffers.
    //
    // buffer_table_ contains pointers to buffers that _will_ be written to by
    // generated code so it would be misleading to make buffer_table_ a `const
    // void**`.
    buffer_table_[arg_index_table_[index]] = const_cast<void*>(data);
  }

  // ------------------------------
  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. Unlike the arg methods,
  // there is no set_resultN_data method. The result buffers are managed
  // internally, and may change after each call to Run.

  // Returns the underlying array of result buffers, where results()[I] is the
  // buffer for the positional result at index I.
  void** results() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_8(mht_8_v, 396, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "results");
 return static_cast<void**>(buffer_table_[result_index_]); }
  const void* const* results() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_9(mht_9_v, 400, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "results");

    return static_cast<const void* const*>(buffer_table_[result_index_]);
  }

  // Profile counters for this XLA computation.
  //
  // When Hlo profiling is enabled (`hlo_profiling_enabled()` return true in
  // this case) these counters are non-null and are automatically populated by
  // `Run`.  The counters can then be pretty-printed using
  // `hlo_profile_printer()`.
  //
  // When Hlo profiling is disabled, this accessor returns null.
  const int64_t* profile_counters() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_10(mht_10_v, 415, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "profile_counters");
 return profile_counters_; }

  // Returns the buffer for the positional result at the given `index`.
  void* result_data(size_t index) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_11(mht_11_v, 421, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "result_data");
 return results()[index]; }
  const void* result_data(size_t index) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_12(mht_12_v, 425, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "result_data");
 return results()[index]; }

  // ------------------------------
  // Methods for extracting optional metadata.

  // Returns true iff data is available for the Lookup{Arg,Variable,Result}Index
  // methods. E.g. the data might not be compiled into the binary for AOT.
  bool HasNameIndices() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_13(mht_13_v, 435, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "HasNameIndices");

    return arg_names_ != nullptr && variable_names_ != nullptr &&
           result_names_ != nullptr;
  }

  // Returns the 0-based index for the argument with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupArgIndex(const string& name) const;

  // Returns the 0-based index for the variable with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupVariableIndex(const string& name) const;

  // Returns the 0-based index for the result with the given `name`.
  // Returns -1 if the name wasn't found, or data isn't available.
  //
  // The index remains constant for every instance of XlaCompiledCpuFunction
  // generated from the same static data, and might not be cheap to determine.
  // Recommended usage is to capture this in a variable for re-use.
  int LookupResultIndex(const string& name) const;

  // Returns the shape of the args and results. May return nullptr if the
  // program shape isn't available.
  const xla::ProgramShapeProto* ProgramShape() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_14(mht_14_v, 469, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "ProgramShape");
 return program_shape_; }

  bool hlo_profiling_enabled() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_15(mht_15_v, 474, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "hlo_profiling_enabled");

    return hlo_profile_printer_data_ != nullptr;
  }
  const xla::HloProfilePrinterData& hlo_profile_printer_data() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_16(mht_16_v, 480, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "hlo_profile_printer_data");

    assert(hlo_profiling_enabled());
    return *hlo_profile_printer_data_;
  }

 protected:
  // ---------------------------------------------------------------------------
  // Accessors for reading from and writing to instances of `StaticData`.
  //
  // Classes generated by tfcompile can call these because the generated classes
  // inherit from `XlaCompiledCpuFunction`.  `XlaJitCompiledCpuFunction` can
  // call these because it is explicitly added as a friend.

  static void set_static_data_raw_function(StaticData* static_data,
                                           RawFunction raw_function) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_17(mht_17_v, 497, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_raw_function");

    static_data->raw_function_ = raw_function;
  }

  static void set_static_data_buffer_infos(
      StaticData* static_data,
      const xla::cpu_function_runtime::BufferInfo* buffer_infos) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_18(mht_18_v, 506, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_buffer_infos");

    static_data->buffer_infos_ = buffer_infos;
  }

  static void set_static_data_num_buffers(StaticData* static_data,
                                          size_t num_buffers) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_19(mht_19_v, 514, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_num_buffers");

    static_data->num_buffers_ = num_buffers;
  }

  static void set_static_data_arg_index_table(StaticData* static_data,
                                              const int32* arg_index_table) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_20(mht_20_v, 522, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_arg_index_table");

    static_data->arg_index_table_ = arg_index_table;
  }

  static void set_static_data_num_args(StaticData* static_data,
                                       int64_t num_args) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_21(mht_21_v, 530, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_num_args");

    static_data->num_args_ = num_args;
  }

  static void set_static_data_num_variables(StaticData* static_data,
                                            int64_t num_variables) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_22(mht_22_v, 538, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_num_variables");

    static_data->num_variables_ = num_variables;
  }

  static void set_static_data_result_index(StaticData* static_data,
                                           size_t result_index) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_23(mht_23_v, 546, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_result_index");

    static_data->result_index_ = result_index;
  }

  static void set_static_data_arg_names(StaticData* static_data,
                                        const char** arg_names) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_24(mht_24_v, 554, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_arg_names");

    static_data->arg_names_ = arg_names;
  }

  static void set_static_data_variable_names(StaticData* static_data,
                                             const char** variable_names) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_25(mht_25_v, 562, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_variable_names");

    static_data->variable_names_ = variable_names;
  }

  static void set_static_data_result_names(StaticData* static_data,
                                           const char** result_names) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_26(mht_26_v, 570, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_result_names");

    static_data->result_names_ = result_names;
  }

  static void set_static_data_program_shape(
      StaticData* static_data, const xla::ProgramShapeProto* program_shape) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_27(mht_27_v, 578, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_program_shape");

    static_data->program_shape_ = program_shape;
  }

  static void set_static_data_hlo_profile_printer_data(
      StaticData* static_data,
      const xla::HloProfilePrinterData* hlo_profile_printer_data) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_28(mht_28_v, 587, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_hlo_profile_printer_data");

    static_data->hlo_profile_printer_data_ = hlo_profile_printer_data;
  }

  static const xla::HloProfilePrinterData*
  get_static_data_hlo_profile_printer_data(StaticData* static_data) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_29(mht_29_v, 595, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "get_static_data_hlo_profile_printer_data");

    return static_data->hlo_profile_printer_data_;
  }

  static void set_static_data_profile_counters_size(
      StaticData* static_data, int64_t profile_counters_size) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_compiled_cpu_functionDTh mht_30(mht_30_v, 603, "", "./tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h", "set_static_data_profile_counters_size");

    static_data->profile_counters_size_ = profile_counters_size;
  }

 private:
  const RawFunction raw_function_;
  const size_t result_index_;

  // Array containing pointers to argument and temp buffers (slots corresponding
  // to constant and on-stack buffers are null).
  void** const buffer_table_;

  // Describes the buffers used by the XLA computation.
  const xla::cpu_function_runtime::BufferInfo* const buffer_infos_;

  // Argument i needs to be placed in buffer_table_[arg_index_to_temp_index_[i]]
  // for XLA generated code to be able to find it.
  const int32* const arg_index_table_;

  // The number of incoming arguments.
  const int32 num_args_;

  // The number of incoming variables.
  const int32 num_variables_;

  // Backing memory for buffer_table_ and args_, the latter depending on
  // AllocMode.
  void* alloc_buffer_table_ = nullptr;

  // Backing memory for profiling counters.
  int64_t* profile_counters_ = nullptr;

  // Options and context passed to the compiled function.
  xla::ExecutableRunOptions run_options_;

  // Optional metadata.
  const char** arg_names_ = nullptr;
  const char** variable_names_ = nullptr;
  const char** result_names_ = nullptr;
  const xla::ProgramShapeProto* program_shape_ = nullptr;
  const xla::HloProfilePrinterData* hlo_profile_printer_data_ = nullptr;

  // Add `XlaJitCompiledCpuFunction` as a friend so that it can access the
  // `set_static_data_*` static methods above.
  friend class XlaJitCompiledCpuFunction;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILED_CPU_FUNCTION_H_
