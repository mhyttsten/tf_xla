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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_testDTcc() {
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

#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.h"

#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace experimental {
namespace {

std::string CompileHloConvAndGetMlir(absl::string_view hlo_text) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPSexperimentalPSconv_emitterPSconv_emitter_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter_test.cc", "CompileHloConvAndGetMlir");

  xla::HloModuleConfig hlo_config;
  VerifiedHloModule hlo_module(
      "Conv", hlo_config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      /*shape_size_function=*/ShapeUtil::ByteSizeOfElements);
  TF_CHECK_OK(hlo_module.ParseHloStringAndVerifyModule(hlo_text));
  xla::HloInstruction* conv =
      hlo_module.entry_computation()->root_instruction();

  mlir::MLIRContext context;
  context.loadDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                      mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));

  mlir::func::FuncOp function =
      EmitConvolutionForwardAsMlir(conv, "Conv", &context).ValueOrDie();

  mlir_module->push_back(function);
  (void)mlir_module->verifyInvariants();

  std::string mlir_text;
  {
    llvm::raw_string_ostream strstream(mlir_text);
    function.print(strstream);
  }
  VLOG(1) << mlir_text;

  {
    mlir::PassManager pm(mlir_module->getContext());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createMemRefToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    CHECK(mlir::succeeded(pm.run(*mlir_module)));
  }

  return mlir_text;
}

// TODO(timshen): integrate this with mlir's testing infrastructure.
TEST(ConvEmitterTest, TestDefault) {
  std::string hlo_text = R"(HloModule TestModule
ENTRY %TestComputation {
  %param_0 = f16[128,4,224,224]{1,3,2,0} parameter(0)
  %param_1 = f16[7,7,64,4]{3,1,0,2} parameter(1)
  ROOT %custom-call.1 = (f16[128,64,112,112]{1,3,2,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=bf01_01oi->bf01, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})";

  std::string expected_mlir_pattern =
      R"(
CHECK: func @Conv(%arg0: memref<128x112x112x64xf16>, %arg1: memref<128x224x224x4xf16>, %arg2: memref<64x7x7x4xf16>) {
CHECK-NEXT:   affine.for %arg3 = 0 to 128 {
CHECK-NEXT:     affine.for %arg4 = 0 to 2 {
CHECK-NEXT:       affine.for %arg5 = 0 to 112 {
CHECK-NEXT:         affine.for %arg6 = 0 to 7 {
CHECK-NEXT:           %0 = memref.alloc() : memref<32x16xf32>
CHECK-NEXT:           affine.for %arg7 = 0 to 32 {
CHECK-NEXT:             affine.for %arg8 = 0 to 16 {
CHECK-NEXT:               %cst = arith.constant 0.000000e+00 : f32
CHECK-NEXT:               affine.store %cst, %0[%arg7, %arg8] : memref<32x16xf32>
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:           affine.for %arg7 = 0 to 1 {
CHECK-NEXT:             affine.for %arg8 = 0 to 7 {
CHECK-NEXT:               affine.for %arg9 = 0 to 7 {
CHECK-NEXT:                 affine.for %arg10 = 0 to 32 {
CHECK-NEXT:                   affine.for %arg11 = 0 to 16 {
CHECK-NEXT:                     affine.for %arg12 = 0 to 4 {
CHECK-NEXT:                       %1 = affine.load %arg1[%arg3, %arg5 * 2 + %arg8 - 3, (%arg6 * 16 + %arg11) * 2 + %arg9 - 3, %arg7 * 4 + %arg12] : memref<128x224x224x4xf16>
CHECK-NEXT:                       %2 = arith.extf %1 : f16 to f32
CHECK-NEXT:                       %3 = affine.load %arg2[%arg4 * 32 + %arg10, %arg8, %arg9, %arg7 * 4 + %arg12] : memref<64x7x7x4xf16>
CHECK-NEXT:                       %4 = arith.extf %3 : f16 to f32
CHECK-NEXT:                       %5 = affine.load %0[%arg10, %arg11] : memref<32x16xf32>
CHECK-NEXT:                       %6 = arith.mulf %2, %4 : f32
CHECK-NEXT:                       %7 = arith.addf %5, %6 : f32
CHECK-NEXT:                       affine.store %7, %0[%arg10, %arg11] : memref<32x16xf32>
CHECK-NEXT:                     }
CHECK-NEXT:                   }
CHECK-NEXT:                 }
CHECK-NEXT:               }
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:           affine.for %arg7 = 0 to 32 {
CHECK-NEXT:             affine.for %arg8 = 0 to 16 {
CHECK-NEXT:               %1 = affine.load %0[%arg7, %arg8] : memref<32x16xf32>
CHECK-NEXT:               %2 = arith.truncf %1 : f32 to f16
CHECK-NEXT:               affine.store %2, %arg0[%arg3, %arg5, %arg6 * 16 + %arg8, %arg4 * 32 + %arg7] : memref<128x112x112x64xf16>
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:         }
CHECK-NEXT:       }
CHECK-NEXT:     }
CHECK-NEXT:   }
CHECK-NEXT:   return
CHECK-NEXT: }
)";

  EXPECT_TRUE(
      RunFileCheck(CompileHloConvAndGetMlir(hlo_text), expected_mlir_pattern)
          .ValueOrDie());
}

}  // namespace
}  // namespace experimental
}  // namespace xla
