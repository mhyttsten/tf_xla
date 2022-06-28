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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc() {
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

#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {

using gpu::GPUModuleOp;

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToNVVMPass
    : public GpuKernelToNVVMPassBase<GpuKernelToNVVMPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_lowering_passes.cc", "getDependentDialects");

    registry.insert<mlir::NVVM::NVVMDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc mht_1(mht_1_v, 227, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_lowering_passes.cc", "runOnOperation");

    GPUModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    mlir::LowerToLLVMOptions llvm_opts(m.getContext(), DataLayout(m));

    LLVMTypeConverter converter(m.getContext(), llvm_opts);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateGpuToNVVMConversionPatterns(converter, patterns);
    populateComplexToLLVMConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

/// A pass that does the final lowering to ROCDL. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class GpuKernelToROCDLPass
    : public GpuKernelToROCDLPassBase<GpuKernelToROCDLPass> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc mht_2(mht_2_v, 256, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_lowering_passes.cc", "getDependentDialects");

    registry.insert<mlir::ROCDL::ROCDLDialect, mlir::LLVM::LLVMDialect>();
  }

 public:
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPSkernel_lowering_passesDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_lowering_passes.cc", "runOnOperation");

    gpu::GPUModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(m.getContext());
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateGpuToROCDLConversionPatterns(converter, patterns,
                                         gpu::amd::Runtime::Unknown);
    populateComplexToLLVMConversionPatterns(converter, patterns);
    ConversionTarget target(getContext());
    configureGpuToROCDLConversionLegality(target);
    if (failed(mlir::applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<GPUModuleOp> > CreateGpuKernelToNvvmPass() {
  return std::make_unique<GpuKernelToNVVMPass>();
}

std::unique_ptr<OperationPass<GPUModuleOp> > CreateGpuKernelToRocdlPass() {
  return std::make_unique<GpuKernelToROCDLPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
