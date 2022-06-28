/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePScommonPStfl_pass_configDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePScommonPStfl_pass_configDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePScommonPStfl_pass_configDTh() {
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


#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"

namespace mlir {
namespace TFL {

// A config that controls which passes get run as part TFLite converter.
struct PassConfig {
  explicit PassConfig(quant::QuantizationSpecs specs)
      : emit_builtin_tflite_ops(true),
        lower_tensor_list_ops(false),
        trim_functions_allowlist({}),
        quant_specs(std::move(specs)),
        form_clusters(false),
        unfold_batch_matmul(true),
        shape_inference(true),
        runtime_verification(true),
        enable_tflite_variables(false),
        disable_variable_freezing(false),
        unfold_large_splat_constant(false),
        guarantee_all_funcs_one_use(false),
        enable_hlo_to_tf_conversion(false),
        enable_dynamic_update_slice(false),
        preserve_assert_op(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePScommonPStfl_pass_configDTh mht_0(mht_0_v, 217, "", "./tensorflow/compiler/mlir/lite/common/tfl_pass_config.h", "PassConfig");
}

  // If `emit_builtin_tflite_ops` is true, TF Lite legalization passes will be
  // added, which produces TF Lite ops.
  bool emit_builtin_tflite_ops;
  // If `lower_tensor_list_ops` is true, tensorlist ops will be lowered to basic
  // TF ops before legalization to TF Lite dialect.
  bool lower_tensor_list_ops;
  // The allowlist of functions that would be preserved after trimming.
  llvm::ArrayRef<std::string> trim_functions_allowlist;
  // All information about quantization.
  quant::QuantizationSpecs quant_specs;
  // If `form_clusters` is true , clusters are formed by grouping consecutive
  // ops of the same device, under a `tf_device.launch` op.
  bool form_clusters;
  // if `unfold_batch_matmul` is true, the tf.BatchMatMul is unfolded to a set
  // of tfl.fully_connected ops.
  bool unfold_batch_matmul;
  // Whether to outline WhileOp at the end of the pipeline.
  bool outline_tf_while = false;
  // Whether to do shape inference.
  bool shape_inference;
  // Whether to do TFLite runtime verification.
  bool runtime_verification;
  // Whether to enable TFLite variables or not, this will allow
  // mutable variables and produce ReadVariable/AssignVariable ops in TFLite.
  bool enable_tflite_variables;
  // Whether to disable the variable freezing pass or not.
  // By default we freeze all variables and disallow mutable variables. When
  // 'enable_tflite_variables' is true then we allow mutable variable only.
  bool disable_variable_freezing;
  // Whether to unfold large splat constant tensors and replace them with
  // fill operation.
  bool unfold_large_splat_constant;
  // Whether to run the `GuaranteeAllFuncsOneUsePass` to ensure each function
  // has a single use.
  bool guarantee_all_funcs_one_use;
  // Whether to enable the hlo to tf conversion.
  bool enable_hlo_to_tf_conversion;
  // Whether to enable to use DynamicUpdateSlice op.
  bool enable_dynamic_update_slice;
  // Whether to preserve AssertOp during legalization.
  bool preserve_assert_op;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const PassConfig& pass_config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePScommonPStfl_pass_configDTh mht_1(mht_1_v, 266, "", "./tensorflow/compiler/mlir/lite/common/tfl_pass_config.h", "operator<<");

  return os << "emit_builtin_tflite_ops: "
            << pass_config.emit_builtin_tflite_ops
            << "\nlower_tensor_list_ops: " << pass_config.lower_tensor_list_ops
            << "\ntrim_functions_allowlist: "
            << absl::StrJoin(pass_config.trim_functions_allowlist.vec(), ",")
            << "\nform_clusters: " << pass_config.form_clusters
            << "\nunfold_batch_matmul: " << pass_config.unfold_batch_matmul
            << "\noutline_tf_while: " << pass_config.outline_tf_while
            << "\nshape_inference: " << pass_config.shape_inference
            << "\nruntime_verification: " << pass_config.runtime_verification
            << "\nenable_tflite_variables: "
            << pass_config.enable_tflite_variables
            << "\ndisable_variable_freezing: "
            << pass_config.disable_variable_freezing
            << "\nunfold_large_splat_constant: "
            << pass_config.unfold_large_splat_constant
            << "\nguarantee_all_funcs_one_use: "
            << pass_config.guarantee_all_funcs_one_use
            << "\nenable_hlo_to_tf_conversion: "
            << pass_config.enable_hlo_to_tf_conversion << "\n";
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_COMMON_TFL_PASS_CONFIG_H_
