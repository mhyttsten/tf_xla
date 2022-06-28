/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_
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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTh {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTh() {
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

#include "absl/types/optional.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

// Base class for handling SPMD expansion of a MLIR TF Operation.
class SPMDExpanderBase {
 public:
  virtual ~SPMDExpanderBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expanderDTh mht_0(mht_0_v, 205, "", "./tensorflow/dtensor/mlir/spmd_expander.h", "~SPMDExpanderBase");
}

  // Converts `op` to a SPMD expanded form. SPMD expansion logic is
  // a function of op type, op output's layout, and layout of op's
  // inputs. Must return the `op` that is expanded as the final return value.
  virtual StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) = 0;

  // Layout propagation functions.
  //
  // During the layout algorithm, for each op output we compute a layout by
  // merging the current layout request from the op producing the output and the
  // layout requests from the ops consuming the output. These merged layouts
  // represent the current state of layouts over the entire mlir module.
  //
  // For an op, if any of the merged layouts for the inputs or output are
  // updated, the ComputeLayoutForward and ComputeLayoutBackward functions will
  // be called with all the updated layout maps populated.
  //
  // ComputeLayoutForward should take the input layouts and determine which
  // output layout these inputs would produce. Likewise, ComputeLayoutBackward
  // should take the output layouts and determine the what layouts to propagate
  // to the inputs.
  //
  // In both cases the functions should choose layouts that reduce the amount of
  // cross device communication for the op.
  //
  // ComputeLayoutForward should not take into account the current output
  // layout(s) when computing the new ones. The merge algorithm will decide what
  // to do. There are only a very few cases where the current output layout may
  // need to propagated again, in which case those ops can override the
  // expanded ComputeLayout* functions. This similarly applies to
  // ComputeLayoutBackward.
  //
  // Note that for some ops, where the input layout does not determine output
  // layout (and visa versa), it is acceptable to either return a replicated
  // layout. E.g. for tf.Fill, ComputeLayoutForward can return a replicated
  // output layout and if a consumer requests a more sharded layout, then the
  // layout algorithm will merge the requests, resulting in the more sharded
  // layout.

  // Computes output layout(s) of `op` based on the current `input_layouts`
  // inferred from inputs of `op`. The `input_layouts` parameter maps input
  // indices to the corresponding layouts. It may be empty if the op has no
  // operands or if no input layouts have been inferred yet.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts);

  // Computes output layout(s) of `op` based on the current `input_layouts` and
  // `output_layouts` inferred from the inputs and outputs of `op`. Both
  // parameters maps input/output indices to the corresponding layouts. Either
  // may be empty.
  //
  // NOTE: The other ComputeLayoutForward function should be preferred since in
  // most cases the output layouts are only computed based on the input layouts.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
      const llvm::DenseMap<int, Layout>& output_layouts);

  // Computes input layout(s) of `op` based on the current `output_layouts`
  // inferred from outputs of `op`. The `output_layouts` parameter maps output
  // indices to the corresponding layouts. It may be empty if the op has no
  // outputs or if no output layouts have been inferred yet.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts);

  // Computes input layout(s) of `op` based on the current `output_layouts` and
  // `input_layouts` inferred from the outputs and inputs of `op`. Both
  // parameters maps input/output indices to the corresponding layouts. Either
  // may be empty.
  //
  // NOTE: The other ComputeLayoutBackward function should be preferred since in
  // most cases the input layouts are only computed based on the output layouts.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
      const llvm::DenseMap<int, Layout>& output_layouts);

  // Run ExpandOp() and set layout from the computed layout from original op.
  // Returns the expanded op in output.
  Status ExpandOpAndSetLayout(mlir::Operation* op, mlir::Operation** output);
};

// Computes the SPMD expansion for `op`.
//
// Prior to this call, all inputs to `op` have been lowered to local operations
// & shapes. The lowered op must emit a type compatible with the local shape.
Status RunSPMDExpansion(mlir::Operation* op, mlir::Operation** output);

// A registry of SPMD expanders. This map is statically stored and initialized
// with all the registered SPMD expanders.
class SPMDExpanderRegistry {
 public:
  ~SPMDExpanderRegistry() = default;

  // A singleton available at startup.
  static SPMDExpanderRegistry* Global();

  // Returns the expansion for the given operation (or nullptr if no expansion
  // has been registered).
  SPMDExpanderBase* GetPropagateFnForOp(mlir::Operation* op);

  // Registers an expander for the provided opName.
  InitOnStartupMarker RegisterPropagateFn(
      std::string opName, std::unique_ptr<SPMDExpanderBase> prop);

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<SPMDExpanderBase>>
      op_to_propagate_fn_map_;
};

#define REGISTER_SPMD(name, op, prop, ...)                     \
  static ::tensorflow::InitOnStartupMarker const spmd_##name = \
      InitOnStartupMarker{}                                    \
      << SPMDExpanderRegistry::Global()->RegisterPropagateFn(  \
             mlir::op ::getOperationName().str(),              \
             absl::make_unique<prop>(__VA_ARGS__))

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_
