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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc() {
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

#include <stdexcept>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/Pattern.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/MathToLibm/MathToLibm.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

constexpr StringRef kTfWrapperLibaryLaunchHelperName =
    "_mlir_ciface_tf_launch_kernel";

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

/// A rewrite patter to convert gpu.launch_func operations into a runtime call
/// for the TensorFlow runtime.
class ConvertLaunchFuncOpToTfRuntimeCallPattern
    : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
 public:
  ConvertLaunchFuncOpToTfRuntimeCallPattern(LLVMTypeConverter &type_converter,
                                            StringRef gpu_binary_annotation)
      : ConvertOpToLLVMPattern<gpu::LaunchFuncOp>(type_converter),
        gpu_binary_annotation_(gpu_binary_annotation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_0(mht_0_v, 234, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "ConvertLaunchFuncOpToTfRuntimeCallPattern");
}

 private:
  Value generateParamsArray(gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
                            OpBuilder &builder) const;
  Value generateKernelNameConstant(StringRef moduleName, StringRef name,
                                   Location loc, OpBuilder &builder) const;

  LogicalResult matchAndRewrite(
      gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override;

  MLIRContext *context_ = &this->getTypeConverter()->getContext();

  Type llvm_void_type_ = LLVM::LLVMVoidType::get(context_);
  Type llvm_pointer_type_ =
      LLVM::LLVMPointerType::get(IntegerType::get(context_, 8));
  Type llvm_pointer_pointer_type_ =
      LLVM::LLVMPointerType::get(llvm_pointer_type_);
  Type llvm_int8_type_ = IntegerType::get(context_, 8);
  Type llvm_int32_type_ = IntegerType::get(context_, 32);
  Type llvm_int64_type_ = IntegerType::get(context_, 64);
  Type llvm_intptr_type_ = IntegerType::get(
      context_, this->getTypeConverter()->getPointerBitwidth(0));

  llvm::SmallString<32> gpu_binary_annotation_;
};

// Creates a struct containing all kernel parameters on the stack and returns
// an array of type-erased pointers to the fields of the struct. The array can
// then be passed to the CUDA / ROCm (HIP) kernel launch calls.
// The generated code is essentially as follows:
//
// %struct = alloca(sizeof(struct { Parameters... }))
// %array = alloca(NumParameters * sizeof(void *))
// for (i : [0, NumParameters))
//   %fieldPtr = llvm.getelementptr %struct[0, i]
//   llvm.store parameters[i], %fieldPtr
//   %elementPtr = llvm.getelementptr %array[i]
//   llvm.store %fieldPtr, %elementPtr
// return %array
Value ConvertLaunchFuncOpToTfRuntimeCallPattern::generateParamsArray(
    gpu::LaunchFuncOp launch_op, OpAdaptor adaptor, OpBuilder &builder) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_1(mht_1_v, 279, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "ConvertLaunchFuncOpToTfRuntimeCallPattern::generateParamsArray");

  auto loc = launch_op.getLoc();
  auto num_kernel_operands = launch_op.getNumKernelOperands();
  auto arguments = getTypeConverter()->promoteOperands(
      loc, launch_op.getOperands().take_back(num_kernel_operands),
      adaptor.operands().take_back(num_kernel_operands), builder);
  auto num_arguments = arguments.size();
  SmallVector<Type, 4> argument_types;
  argument_types.reserve(num_arguments);
  for (auto argument : arguments) argument_types.push_back(argument.getType());
  auto struct_type = LLVM::LLVMStructType::getNewIdentified(
      context_, StringRef(), argument_types);
  auto one = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type_,
                                              builder.getI32IntegerAttr(1));
  auto struct_ptr = builder.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(struct_type), one, /*alignment=*/0);
  auto array_size = builder.create<LLVM::ConstantOp>(
      loc, llvm_int32_type_, builder.getI32IntegerAttr(num_arguments));
  auto array_ptr = builder.create<LLVM::AllocaOp>(
      loc, llvm_pointer_pointer_type_, array_size, /*alignment=*/0);
  auto zero = builder.create<LLVM::ConstantOp>(loc, llvm_int32_type_,
                                               builder.getI32IntegerAttr(0));
  for (auto en : llvm::enumerate(arguments)) {
    auto index = builder.create<LLVM::ConstantOp>(
        loc, llvm_int32_type_, builder.getI32IntegerAttr(en.index()));
    auto field_ptr = builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(argument_types[en.index()]), struct_ptr,
        ArrayRef<Value>{zero, index.getResult()});
    builder.create<LLVM::StoreOp>(loc, en.value(), field_ptr);
    auto element_ptr = builder.create<LLVM::GEPOp>(
        loc, llvm_pointer_pointer_type_, array_ptr, index.getResult());
    auto casted =
        builder.create<LLVM::BitcastOp>(loc, llvm_pointer_type_, field_ptr);
    builder.create<LLVM::StoreOp>(loc, casted, element_ptr);
  }
  return array_ptr;
}

// Emits LLVM IR to launch a kernel function. Expects the module that contains
// the compiled kernel function as a cubin in the 'nvvm.cubin' attribute, or a
// hsaco in the 'rocdl.hsaco' attribute of the kernel function in the IR.
//
// %0 = call %binarygetter
// %1 = <pointer to kernel function name>
// %2 = <see generateParamsArray>
// call %tfLaunchKernel(%ctx, %0, %1, <launch_op operands 0..5>, %2)
LogicalResult ConvertLaunchFuncOpToTfRuntimeCallPattern::matchAndRewrite(
    gpu::LaunchFuncOp launch_op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_2(mht_2_v, 330, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "ConvertLaunchFuncOpToTfRuntimeCallPattern::matchAndRewrite");

  if (!launch_op.asyncDependencies().empty() || launch_op.asyncToken()) {
    return rewriter.notifyMatchFailure(
        launch_op, "Cannot convert with async dependency or result.");
  }

  Location loc = launch_op.getLoc();

  // Create an LLVM global with CUBIN extracted from the kernel annotation and
  // obtain a pointer to the first byte in it.
  auto kernel_module = SymbolTable::lookupNearestSymbolFrom<gpu::GPUModuleOp>(
      launch_op, launch_op.getKernelModuleName());
  assert(kernel_module && "expected a kernel module");

  auto binary_attr =
      kernel_module->getAttrOfType<StringAttr>(gpu_binary_annotation_);
  if (!binary_attr) {
    kernel_module.emitOpError()
        << "missing " << gpu_binary_annotation_ << " attribute";
    return failure();
  }

  // Create a global for the module blob.
  SmallString<128> name_buffer(kernel_module.getName());
  name_buffer.append("_blob");
  Value module_blob =
      LLVM::createGlobalString(loc, rewriter, name_buffer.str(),
                               binary_attr.getValue(), LLVM::Linkage::Internal);

  // Make sure the trailing zero is included in the constant.
  auto kernel_name = launch_op.getKernelName().getValue();
  SmallString<128> kernel_name_buffer(kernel_name);
  kernel_name_buffer.push_back('\0');

  // Create a global for the kernel name.
  SmallString<128> kernel_name_global_name_buffer;
  auto kernel_name_global_name =
      (kernel_module.getName() + "_" + kernel_name + "_kernel_name")
          .toStringRef(kernel_name_global_name_buffer);
  auto kernel_name_global =
      LLVM::createGlobalString(loc, rewriter, kernel_name_global_name,
                               kernel_name_buffer, LLVM::Linkage::Internal);

  // The TensorFlow OpKernelContext is the first argument of the surrounding
  // LLVMFunc.
  Value context_arg =
      launch_op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);
  auto kernel_params = generateParamsArray(launch_op, adaptor, rewriter);

  auto libraryLaunchNameAttr =
      mlir::StringAttr::get(loc.getContext(), kTfWrapperLibaryLaunchHelperName);
  auto function = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
      launch_op, libraryLaunchNameAttr);
  if (!function) {
    PatternRewriter::InsertionGuard guard(rewriter);
    auto function_type = LLVM::LLVMFunctionType::get(
        llvm_void_type_,
        {
            llvm_pointer_type_,         /* void* context */
            llvm_pointer_type_,         /* void* module_blob */
            llvm_pointer_type_,         /* void* function_name */
            llvm_intptr_type_,          /* intptr_t grid_x_dim */
            llvm_intptr_type_,          /* intptr_t grid_y_dim */
            llvm_intptr_type_,          /* intptr_t grid_z_dim */
            llvm_intptr_type_,          /* intptr_t block_x_dim */
            llvm_intptr_type_,          /* intptr_t block_y_dim */
            llvm_intptr_type_,          /* intptr_t block_z_dim */
            llvm_pointer_pointer_type_, /* void **kernel_params */
        });
    rewriter.setInsertionPointToStart(
        launch_op->getParentOfType<ModuleOp>().getBody());
    function = rewriter.create<LLVM::LLVMFuncOp>(
        loc, kTfWrapperLibaryLaunchHelperName, function_type);
  }
  rewriter.create<LLVM::CallOp>(
      loc, TypeRange(), mlir::SymbolRefAttr::get(function),

      ArrayRef<Value>{
          context_arg, module_blob, kernel_name_global, adaptor.gridSizeX(),
          adaptor.gridSizeY(), adaptor.gridSizeZ(), adaptor.blockSizeX(),
          adaptor.blockSizeY(), adaptor.blockSizeZ(), kernel_params});

  rewriter.eraseOp(launch_op);
  return success();
}

class TFKernelToLLVMPass : public TFKernelToLLVMPassBase<TFKernelToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_3(mht_3_v, 420, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "getDependentDialects");

    registry.insert<LLVM::LLVMDialect>();
  }

 public:
  explicit TFKernelToLLVMPass(StringRef blob_annotation) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_4(mht_4_v, 428, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "TFKernelToLLVMPass");

    if (!blob_annotation.empty()) {
      blob_annotation_ = blob_annotation.str();
    }
  }

  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStransformsPStf_kernel_to_llvm_passDTcc mht_5(mht_5_v, 437, "", "./tensorflow/compiler/mlir/tools/kernel_gen/transforms/tf_kernel_to_llvm_pass.cc", "runOnOperation");

    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext *ctx = m.getContext();
    LLVMTypeConverter type_converter(ctx);
    type_converter.addConversion([&](tf_framework::OpKernelContextType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });
    type_converter.addConversion([&](tf_framework::JITCallableType type) {
      return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    });

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    arith::populateArithmeticExpandOpsPatterns(patterns);
    memref::populateExpandOpsPatterns(patterns);
    arith::populateArithmeticToLLVMConversionPatterns(type_converter, patterns);
    populateMemRefToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLLVMConversionPatterns(type_converter, patterns);
    populateFuncToLLVMConversionPatterns(type_converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(type_converter, patterns);
    populateComplexToLLVMConversionPatterns(type_converter, patterns);
    populateVectorToLLVMConversionPatterns(type_converter, patterns);
    populateMathToLibmConversionPatterns(patterns, 0);
    tf_framework::PopulateTFFrameworkToLLVMConversionPatterns(&type_converter,
                                                              &patterns);
    patterns.add<ConvertLaunchFuncOpToTfRuntimeCallPattern>(type_converter,
                                                            blob_annotation_);
    //  Set target.
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<
        arith::ArithmeticDialect, func::FuncDialect, complex::ComplexDialect,
        gpu::GPUDialect, tf_framework::TFFrameworkDialect, math::MathDialect>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp, gpu::GPUModuleOp>();
    // Do not look into gpu modules, only consider host-side.
    target.markOpRecursivelyLegal<gpu::GPUModuleOp>();
    // Unrealized conversion casts are cleaned up by a separate pass.
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Finally, strip the GPU modules, as they are no longer needed.
    for (auto op : llvm::make_early_inc_range(m.getOps<gpu::GPUModuleOp>())) {
      op.erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateTFKernelToLLVMPass(
    StringRef blob_annotation) {
  return std::make_unique<TFKernelToLLVMPass>(blob_annotation);
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
