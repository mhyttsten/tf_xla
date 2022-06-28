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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc() {
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
// Provide helper routine for obtaining  gpu target information useful
// for llvm IR contruction.

#include "tensorflow/compiler/xla/service/gpu/target_util.h"

#include "absl/strings/str_cat.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/MDBuilder.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {
namespace {
// Utility functions to obtain NVPTX/AMDGPU specific information.
using absl::StrCat;

// Wrapper structure for carrying llvm intrinsic ids for NVPTX/AMDGPU platforms.
// On AMDGPU, some of these operations are made as device functions instead of
// intrinsics. Therefore a variant type is used to wrap the lambda to call
// those device functions.
struct TargetIntrinsics {
  llvm::Intrinsic::ID nvptx_intrinsic;
  absl::variant<llvm::Intrinsic::ID,
                std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>
      amdgpu_intrinsic_or_function;
};

// Gets the llvm intrinsic ids on different platforms (NVPTX, AMDGPU)
// corresponding to the give TargetIntrinsicID.
struct TargetIntrinsics GetIntrinsic(TargetIntrinsicID intrin) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "GetIntrinsic");

  switch (intrin) {
    case TargetIntrinsicID::kThreadIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
              llvm::Intrinsic::amdgcn_workitem_id_x};
    }
    case TargetIntrinsicID::kThreadIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
              llvm::Intrinsic::amdgcn_workitem_id_y};
    }
    case TargetIntrinsicID::kThreadIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
              llvm::Intrinsic::amdgcn_workitem_id_z};
    }
    case TargetIntrinsicID::kBlockIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
              llvm::Intrinsic::amdgcn_workgroup_id_x};
    }
    case TargetIntrinsicID::kBlockIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
              llvm::Intrinsic::amdgcn_workgroup_id_y};
    }
    case TargetIntrinsicID::kBlockIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
              llvm::Intrinsic::amdgcn_workgroup_id_z};
    }
    case TargetIntrinsicID::kBarrierId: {
      return {llvm::Intrinsic::nvvm_barrier0,
              llvm::Intrinsic::amdgcn_s_barrier};
    }
    case TargetIntrinsicID::kBlockDimx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(0)}, {U32}, U64, {},
                                              b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(1)}, {U32}, U64, {},
                                              b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(2)}, {U32}, U64, {},
                                              b_);
              }};
    }
  }
}

// Wrapper structure for carrying math functions for NVPTX/AMDGPU platforms.
struct TargetDeviceFunction {
  const std::string nvptx_root;
  const std::string amdgpu_root;
};

// Gets the device function name on different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetDeviceFunctionID.
struct TargetDeviceFunction GetDeviceFunctionRoot(
    TargetDeviceFunctionID func_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_1(mht_1_v, 285, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "GetDeviceFunctionRoot");

  switch (func_id) {
    case TargetDeviceFunctionID::kAtan2: {
      return {"__nv_atan2", "__ocml_atan2"};
    }
    case TargetDeviceFunctionID::kCos: {
      return {"__nv_cos", "__ocml_cos"};
    }
    case TargetDeviceFunctionID::kErfcinv: {
      return {"__nv_erfcinv", "__ocml_erfcinv"};
    }
    case TargetDeviceFunctionID::kExp: {
      return {"__nv_exp", "__ocml_exp"};
    }
    case TargetDeviceFunctionID::kExpm1: {
      return {"__nv_expm1", "__ocml_expm1"};
    }
    case TargetDeviceFunctionID::kFmod: {
      return {"__nv_fmod", "__ocml_fmod"};
    }
    case TargetDeviceFunctionID::kHypot: {
      return {"__nv_hypot", "__ocml_hypot"};
    }
    case TargetDeviceFunctionID::kLog: {
      return {"__nv_log", "__ocml_log"};
    }
    case TargetDeviceFunctionID::kLog1p: {
      return {"__nv_log1p", "__ocml_log1p"};
    }
    case TargetDeviceFunctionID::kPow: {
      return {"__nv_pow", "__ocml_pow"};
    }
    case TargetDeviceFunctionID::kRound: {
      return {"__nv_round", "__ocml_round"};
    }
    case TargetDeviceFunctionID::kRsqrt: {
      return {"__nv_rsqrt", "__ocml_rsqrt"};
    }
    case TargetDeviceFunctionID::kSin: {
      return {"__nv_sin", "__ocml_sin"};
    }
    case TargetDeviceFunctionID::kSqrt: {
      return {"__nv_sqrt", "__ocml_sqrt"};
    }
    case TargetDeviceFunctionID::kTanh: {
      return {"__nv_tanh", "__ocml_tanh"};
    }
  }
}
}  // namespace

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_2(mht_2_v, 341, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "ObtainDeviceFunctionName");

  // The device math functions differentiate between "double" and "float" by
  // appending a double or float specific suffix to a root name. The suffix and
  // the root name are specific to the target.
  llvm::Triple target_triple =
      llvm::Triple(b->GetInsertBlock()->getModule()->getTargetTriple());
  struct TargetDeviceFunction gpu_root_names = GetDeviceFunctionRoot(func_id);
  if (target_triple.isNVPTX()) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.nvptx_root, "f");
    } else if (output_type == F64) {
      return gpu_root_names.nvptx_root;
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.amdgpu_root, "_f32");
    } else if (output_type == F64) {
      return StrCat(gpu_root_names.amdgpu_root, "_f64");
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

llvm::CallInst* EmitDeviceFunctionCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    llvm::IRBuilder<>* b, absl::string_view name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("callee_name: \"" + callee_name + "\"");
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_3(mht_3_v, 378, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "EmitDeviceFunctionCall");

  std::vector<llvm::Type*> ir_input_types;
  llvm::Module* module = b->GetInsertBlock()->getModule();
  for (PrimitiveType input_type : input_types) {
    ir_input_types.push_back(
        llvm_ir::PrimitiveTypeToIrType(input_type, module));
  }
  llvm::FunctionType* callee_type = llvm::FunctionType::get(
      llvm_ir::PrimitiveTypeToIrType(output_type, module),  // Return type.
      ir_input_types,                                       // Parameter types.
      false);  // No variadic arguments.

  // Declares the callee if it is not declared already.
  llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
      b->GetInsertBlock()
          ->getModule()
          ->getOrInsertFunction(callee_name, callee_type)
          .getCallee());

  for (auto attribute : attributes) {
    callee->addFnAttr(attribute);
  }

  return b->CreateCall(callee, llvm_ir::AsArrayRef(operands), name.data());
}

llvm::CallInst* EmitCallToTargetIntrinsic(
    TargetIntrinsicID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_4(mht_4_v, 409, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "EmitCallToTargetIntrinsic");

  llvm::Module* module = b->GetInsertBlock()->getModule();
  struct TargetIntrinsics gpu_intrinsic_id = GetIntrinsic(intrinsic_id);
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  llvm::Intrinsic::ID llvm_intrinsic_id = llvm::Intrinsic::not_intrinsic;
  if (target_triple.isNVPTX()) {
    llvm_intrinsic_id = gpu_intrinsic_id.nvptx_intrinsic;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    llvm::Intrinsic::ID* llvm_intrinsic_id_ptr =
        absl::get_if<llvm::Intrinsic::ID>(
            &gpu_intrinsic_id.amdgpu_intrinsic_or_function);
    if (llvm_intrinsic_id_ptr) {
      llvm_intrinsic_id = *llvm_intrinsic_id_ptr;
    } else {
      std::function<llvm::CallInst*(llvm::IRBuilder<>*)>* builder_func =
          absl::get_if<std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>(
              &gpu_intrinsic_id.amdgpu_intrinsic_or_function);
      return (*builder_func)(b);
    }
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }

  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
}

void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPStarget_utilDTcc mht_5(mht_5_v, 441, "", "./tensorflow/compiler/xla/service/gpu/target_util.cc", "AnnotateFunctionAsGpuKernel");

  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    // Add the declaration of this kernel to llvm.nvvm.annotations so that NVPTX
    // treats function as a CUDA kernel.
    llvm::LLVMContext& context = module->getContext();
    llvm::NamedMDNode* nvvm_annotations_node =
        module->getOrInsertNamedMetadata("nvvm.annotations");
    nvvm_annotations_node->addOperand(llvm::MDNode::get(
        context, {llvm::ConstantAsMetadata::get(func),
                  llvm::MDString::get(context, "kernel"),
                  llvm::ConstantAsMetadata::get(b->getInt32(1))}));

  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    // Attach information so AMDGPU can recognize function as a AMDGPU kernel.
    func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace gpu
}  // namespace xla
