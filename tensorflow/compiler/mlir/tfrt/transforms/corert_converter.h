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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_CORERT_CONVERTER_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_CORERT_CONVERTER_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh() {
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

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {

struct ParseDeviceNameResult {
  std::string device_type;
  std::string device_name;
  std::string op_handler_name;
};

// A helper class for converting CoreRT types and attributes.
class CoreRTConverter : public mlir::TypeConverter {
 public:
  CoreRTConverter(
      mlir::MLIRContext *context,
      const mlir::TF::SideEffectAnalysis::Info *side_effect_analysis);
  // Materialize all derived attributes. Note that this is only needed by
  // CoreRT ops and fallback ops.
  void MaterializeDerivedAttributes(mlir::Operation *op);

  bool IsSupportedNumericDType(mlir::Type type) const;

  // Create a single attribute that contains the named attribute lists. It is an
  // array of pairs. The key must be a string attribute, and the value can be
  // any attribute that is supported by CoreRuntime.
  mlir::ArrayAttr CreateOpAttrs(llvm::ArrayRef<mlir::NamedAttribute> attrs);

  // Similar to CreateOpAttrs, create a single attribute that contains the
  // named attribute lists, which is an array of pairs, with keys and values
  // both being string attributes. The values represent function names.
  // This method also populates a vector of attribute keys to be removed.
  mlir::ArrayAttr CreateOpFuncAttrs(
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::SmallVector<mlir::StringAttr, 4> *func_attr_keys);

  // Parse the device name of `op` to TFRT's device name. For example, "/CPU:0"
  // will be parsed as "cpu". Return None if no device is assigned.
  llvm::Optional<ParseDeviceNameResult> ParseDeviceName(
      llvm::StringRef device_name) const;
  llvm::Optional<ParseDeviceNameResult> ParseDeviceName(
      mlir::Operation *op) const;

  // Convert the device name in a TF op to a op_handler value produced by the
  // corresponding GetOpHandler in the current block. If there does not exist
  // one, insert a GetOpHandler to the beginning of the block and return the
  // device value.
  mlir::Value ConvertOpHandler(mlir::Operation *op, llvm::StringRef device_name,
                               mlir::ConversionPatternRewriter *rewriter);

  // Get a DistributedContext value to be used by the given op. The
  // DistributedContext value should be shared by all operations in the body
  // of the same FuncOp. If there does not exist one, insert a
  // GetDistributedContext op right before the given op and return the result
  // value.
  mlir::Value GetDistributedContext(mlir::Operation *op,
                                    mlir::ConversionPatternRewriter *rewriter);

  // Get a RemoteChainManager value to be used by the given op. The
  // RemoteChainManager value should be shared by all operations in the body
  // of the same FuncOp. If there does not exist one, insert a
  // tfrt_dist.test_create_remote_chain_manager op right before the given op and
  // return the result value.
  mlir::Value GetRemoteChainManager(mlir::Operation *op,
                                    mlir::ConversionPatternRewriter *rewriter);

  // Get a TaskHandle value with the given task name. If the TaskHandle value
  // has already been created for the given task name within the same FuncOp,
  // return this TaskHandle value. Otherwise, insert a tfrt_dist.get_task_handle
  // op right before the given op and return the result value.
  mlir::Value GetTaskHandle(mlir::Operation *op, StringRef task_name,
                            mlir::ConversionPatternRewriter *rewriter);

  // Any local operation which uses any result of the `op` should depend on the
  // given `chain`.
  void RegisterLocalSideEffectChain(mlir::Operation *op, mlir::Value chain) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_0(mht_0_v, 270, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "RegisterLocalSideEffectChain");

    local_side_effect_chains_[op] = chain;
  }

  // Return a local chain for side effects for `op`. If there are multiple
  // chains, a merge_chains kernel will be inserted and the merged chain will be
  // returned.
  mlir::Value GetLocalSideEffectChain(
      mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter);

  // Return a remote chain for side effects for `op`.
  mlir::Value GetRemoteSideEffectChain(
      mlir::Operation *op, StringRef remote_host,
      mlir::ConversionPatternRewriter *rewriter);

  mlir::Type op_handler_type() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_1(mht_1_v, 288, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "op_handler_type");

    return builder_.getType<::tfrt::corert::OpHandlerType>();
  }

  mlir::Type tensor_handle_type() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_2(mht_2_v, 295, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "tensor_handle_type");

    return builder_.getType<::tfrt::corert::TensorHandleType>();
  }

  mlir::Type chain_type() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_3(mht_3_v, 302, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "chain_type");

    return builder_.getType<::tfrt::compiler::ChainType>();
  }

  mlir::Type distributed_context_type() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_4(mht_4_v, 309, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "distributed_context_type");

    return builder_.getType<::tfrt::dist::DistributedContextType>();
  }

  mlir::Builder &builder() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_5(mht_5_v, 316, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "builder");
 return builder_; }

 private:
  // TODO(chky): attributes "_output_shapes" should be removed by any tool that
  // generates TF MLIR dialect, as they are not used by CoreRuntime. Remove this
  // filtering logic once unused attributes are cleaned up in the upper layer.
  bool IsUnusedAttribute(llvm::StringRef name) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTh mht_6(mht_6_v, 325, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h", "IsUnusedAttribute");

    // NOTE: attributes "f.*" are function attribute related and
    // are added during importing graph to MLIR TF Executor dialect. These
    // attributes are not actually used by TF ops with function attributes.
    // TODO(b/180399811): Re-evaluate the usage of these attributes.
    return name == "_output_shapes" || name.contains("f.");
  }

  // Returns the converted attribute in TFRT dialect. If the conversion fails,
  // returns a null attribute instead.
  mlir::Attribute ConvertAttribute(mlir::Attribute attr);

  mlir::TypeAttr ConvertTypeAttribute(mlir::TypeAttr type_attr);

  mlir::StringAttr ConvertSymbolAttrToStringAttr(
      mlir::FlatSymbolRefAttr symbol_attr);

  mlir::Builder builder_;

  const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis_;

  llvm::DenseMap<mlir::Operation *, mlir::Value> local_side_effect_chains_;
  llvm::DenseMap<mlir::Operation *, mlir::Value> distributed_context_by_func_;
  llvm::DenseMap<mlir::Operation *, mlir::Value> remote_chain_mgr_by_func_;
  llvm::DenseMap<mlir::Operation *, llvm::StringMap<mlir::Value>>
      task_handles_by_func_;
  llvm::StringMap<mlir::Value> op_handler_by_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_CORERT_CONVERTER_H_
