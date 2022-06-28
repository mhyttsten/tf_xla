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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc() {
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

// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pattern to lower lmhlo ops with help of the ir emitter to gpu device code
// and gpu dialect ops (gpu.launch_func and gpu.memcpy).

#include <iterator>
#include <numeric>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_helper.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime

namespace tensorflow {

using mlir::ArrayRef;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Value;
using mlir::memref::GetGlobalOp;
using xla::gpu::DeviceToDeviceCopyThunk;
using xla::gpu::IrEmitterContext;
using xla::gpu::IrEmitterUnnested;
using xla::gpu::KernelThunk;
using xla::gpu::Thunk;
using xla::gpu::ThunkSequence;
using ConstantInfo = xla::gpu::GpuExecutable::ConstantInfo;

namespace {

// Replaces lmhlo ops within a module with gpu.launch_func and gpu.memcpy ops.
struct KernelOpsPattern : mlir::OpRewritePattern<mlir::ModuleOp> {
  using OpRewritePattern<mlir::ModuleOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const override;
};

struct RewriteData {
  Operation* op;
  mlir::SmallVector<Value, 4> arguments;
  std::vector<xla::BufferAllocation> allocations;
  std::unique_ptr<ThunkSequence> thunks;
  std::vector<ConstantInfo> constants;
  std::string gpu_module_data;
};

}  // namespace

static llvm::Error MakeError(llvm::StringRef message) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_0(mht_0_v, 262, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "MakeError");

  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}
static llvm::Error MakeError(xla::Status status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_1(mht_1_v, 268, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "MakeError");

  return MakeError(status.error_message());
}

// Clones `op` into a function within a module with `arguments`.
// The `get_global_ops` are the def ops of `arguments`, or null otherwise.
static std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, mlir::func::FuncOp>
CloneToModule(Operation* op, mlir::ValueRange arguments,
              mlir::MutableArrayRef<GetGlobalOp> get_global_ops) {
  auto loc = op->getLoc();
  auto* context = op->getContext();
  mlir::OpBuilder builder(context);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      builder.create<mlir::ModuleOp>(loc);
  builder.setInsertionPointToEnd(module_op->getBody());
  // Clone and annotate the memref.global ops that the memref.get_global ops
  // refer to. The lmhlo.alloc index refers to one of the function arguments.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    Operation* global_op = mlir::SymbolTable::lookupNearestSymbolFrom(
        pair.value(), pair.value().nameAttr());
    auto attr = builder.getIndexAttr(pair.index());
    builder.clone(*global_op)->setAttr("lmhlo.alloc", attr);
  }

  auto func_type = builder.getType<mlir::FunctionType>(
      mlir::TypeRange(arguments), mlir::TypeRange());
  auto func_name = op->getParentOfType<mlir::func::FuncOp>().getName();
  auto func_op = builder.create<mlir::func::FuncOp>(loc, func_name, func_type);
  // Annotate the function arguments if they refer to a memref.global op.
  for (auto pair : llvm::enumerate(get_global_ops)) {
    if (!pair.value()) continue;
    auto attr = builder.getStringAttr(pair.value().name());
    func_op.setArgAttr(pair.index(), "lmhlo.constant_name", attr);
  }
  func_op.setPublic();

  builder.setInsertionPointToEnd(func_op.addEntryBlock());
  mlir::BlockAndValueMapping mapping;
  for (const auto& pair : llvm::zip_first(arguments, func_op.getArguments()))
    mapping.map(std::get<0>(pair), std::get<1>(pair));
  // Clone the memref.get_global ops.
  for (auto get_global_op : get_global_ops) {
    if (!get_global_op) continue;
    mapping.map(get_global_op, builder.clone(*get_global_op)->getResult(0));
  }
  auto* clone = builder.clone(*op, mapping);
  auto name_loc = mlir::NameLoc::get(builder.getStringAttr(func_name));
  clone->setLoc(mlir::FusedLoc::get(context, {loc, name_loc}));
  builder.create<mlir::lmhlo::TerminatorOp>(loc);

  return std::make_tuple(std::move(module_op), func_op);
}

// Converts the argument's shaped types into buffer allocations.
static llvm::Expected<std::vector<xla::BufferAllocation>> GetAllocations(
    ArrayRef<Value> arguments, ArrayRef<GetGlobalOp> get_global_ops) {
  std::vector<xla::BufferAllocation> allocations;
  allocations.reserve(arguments.size());
  for (Value argument : arguments) {
    mlir::ShapedType type = argument.getType().dyn_cast<mlir::ShapedType>();
    if (!type || !type.hasStaticShape())
      return MakeError("Expected static shapes");
    auto element_size_bytes = xla::GetElementTypeBytes(type.getElementType());
    if (!element_size_bytes.ok()) return MakeError(element_size_bytes.status());
    size_t size = *element_size_bytes * type.getNumElements();
    allocations.emplace_back(allocations.size(), size, 0);
  }
  for (auto pair : llvm::zip_first(allocations, get_global_ops))
    std::get<0>(pair).set_constant(std::get<1>(pair));
  return allocations;
}

// Emits thunks and an llvm device code module for the given func_op.
static llvm::Expected<
    std::tuple<std::unique_ptr<ThunkSequence>, std::vector<ConstantInfo>>>
Emit(mlir::func::FuncOp func_op,
     absl::Span<const xla::BufferAllocation> allocations,
     const stream_executor::CudaComputeCapability& cuda_compute_capability,
     const xla::HloModuleConfig& hlo_module_config, llvm::Module* llvm_module) {
  // Hardcoded values for now...
  const char target_triple[] = "nvptx64-nvidia-cuda";
  const char data_layout[] = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";
  const char platform_name[] = "CUDA";
  xla::gpu::GpuDeviceInfo gpu_device_info = {};
  gpu_device_info.threads_per_block_limit = 1024;
  gpu_device_info.threads_per_warp = 32;
  gpu_device_info.shared_memory_per_block = 49152;  // static shmem limit.
  // Should be 1024 for sm7.5, 1536 for sm8.6. This results in more blocks than
  // SMs on those architectures, but doesn't hit any resource limit.
  gpu_device_info.threads_per_core_limit = 2048;
  // This is higher than any SKU, resulting in more blocks than SMs.
  gpu_device_info.core_count = 128;
  gpu_device_info.block_dim_limit_x = 2147483647;
  gpu_device_info.block_dim_limit_y = 65535;
  gpu_device_info.block_dim_limit_z = 65535;

  llvm_module->setTargetTriple(target_triple);
  llvm_module->setDataLayout(data_layout);

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability,
      stream_executor::RocmComputeCapability("gfx000"), func_op->getContext(),
      llvm_module);

  ir_emitter_context.set_allocations(allocations);

  auto ir_emitter =
      IrEmitterUnnested::Create(hlo_module_config, &ir_emitter_context);
  if (!ir_emitter.ok()) return MakeError(ir_emitter.status());

  auto emit_status = (*ir_emitter)->EmitLmhloRegion(&func_op.getBody());
  if (!emit_status.ok()) return MakeError(emit_status);

  return std::make_tuple((*ir_emitter)->ConsumeThunkSequence(),
                         std::move(ir_emitter_context.constants()));
}

// Returns the data to rewrite op without changing the IR.
static llvm::Expected<RewriteData> Match(Operation* op) {
  mlir::SmallVector<Value, 4> arguments = op->getOperands();
  mlir::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  llvm::copy(captures, std::back_inserter(arguments));

  // Collect arguments that are defined by a memref.get_global op. The
  // created module's annotations make the ir emitter recognize them as
  // constants.
  SmallVector<GetGlobalOp, 4> get_global_ops;
  get_global_ops.reserve(arguments.size());
  llvm::transform(
      arguments, std::back_inserter(get_global_ops),
      [](Value argument) { return argument.getDefiningOp<GetGlobalOp>(); });

  auto allocations = GetAllocations(arguments, get_global_ops);
  if (!allocations) return allocations.takeError();
  auto module_op = CloneToModule(op, arguments, get_global_ops);

  xla::HloModuleConfig hlo_module_config;
  xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
  hlo_module_config.set_debug_options(options);
  stream_executor::CudaComputeCapability cuda_compute_capability = {5, 2};
  llvm::LLVMContext llvm_context;
  auto llvm_module = std::make_unique<llvm::Module>("", llvm_context);

  auto emit_result =
      Emit(std::get<mlir::func::FuncOp>(module_op), *allocations,
           cuda_compute_capability, hlo_module_config, llvm_module.get());
  if (!emit_result) return emit_result.takeError();
  auto thunks = std::move(std::get<0>(*emit_result));
  auto constants = std::move(std::get<1>(*emit_result));
  // Inline sequential thunks into the `thunks` vector.
  for (auto it = thunks->begin(); it != thunks->end();) {
    if (it->get()->kind() == Thunk::kSequential) {
      auto sequence = std::move(
          static_cast<xla::gpu::SequentialThunk*>(it->get())->thunks());
      it = thunks->erase(it);
      it = thunks->insert(it, std::make_move_iterator(sequence.begin()),
                          std::make_move_iterator(sequence.end()));
    } else {
      ++it;
    }
  }
  if (!llvm::all_of(*thunks, [](const auto& thunk) {
        Thunk::Kind kinds[] = {Thunk::kKernel, Thunk::kCopy,
                               Thunk::kMemset32BitValue, Thunk::kMemzero};
        auto equal = [&](Thunk::Kind kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_2(mht_2_v, 439, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");
 return thunk->kind() == kind; };
        return llvm::any_of(kinds, equal);
      })) {
    return MakeError("Expected only kernel, copy, memset, and memzero thunks");
  }

  auto libdevice_dir = xla::gpu::GetLibdeviceDir(hlo_module_config);
  auto ptx =
      xla::gpu::nvptx::CompileToPtx(llvm_module.get(), cuda_compute_capability,
                                    hlo_module_config, libdevice_dir);
  if (!ptx.ok()) return MakeError(ptx.status());

  return RewriteData{op,
                     std::move(arguments),
                     std::move(*allocations),
                     std::move(thunks),
                     std::move(constants),
                     std::move(*ptx)};
}

// Replaces op with gpu.launch_func and gpu.memcpy ops.
static void Rewrite(Operation* op, mlir::PatternRewriter& rewriter,
                    mlir::SymbolTable& symbol_table, ArrayRef<Value> arguments,
                    ThunkSequence* thunks, ArrayRef<ConstantInfo> constants,
                    mlir::StringRef gpu_module_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_3(mht_3_v, 466, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "Rewrite");

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto loc = op->getLoc();

  rewriter.setInsertionPoint(op->getParentOfType<mlir::func::FuncOp>());
  auto gpu_module = rewriter.create<mlir::gpu::GPUModuleOp>(loc, "gpu_module");
  symbol_table.insert(gpu_module);
  gpu_module->setAttr(tfrt::gpu::getGpuBinaryAttrName(),
                      rewriter.getStringAttr(gpu_module_data));

  // Annotate memref.global ops with the gpu.module symbol, and annotate the
  // gpu.module op with memref.global symbols which require initialization.
  SmallVector<mlir::Attribute, 4> const_attrs;
  for (const auto& constant : constants) {
    auto global_op = mlir::SymbolTable::lookupNearestSymbolFrom(
        op, rewriter.getStringAttr(constant.symbol_name));
    if (!global_op) {
      LOG(WARNING) << "memref.global op not found for constant. Possibly "
                   << "unused (spurious) constant.";
      continue;
    }
    global_op->setAttr(tfrt::gpu::getGpuModuleAttrName(),
                       mlir::SymbolRefAttr::get(gpu_module));
    if (!constant.content.empty())
      const_attrs.emplace_back(mlir::SymbolRefAttr::get(global_op));
  }
  if (!const_attrs.empty()) {
    gpu_module->setAttr(tfrt::gpu::getGpuConstantsAttrName(),
                        rewriter.getArrayAttr(const_attrs));
  }

  for (const auto& thunk : *thunks) {
    if (thunk->kind() == Thunk::kCopy) {
      const auto* copy_thunk =
          static_cast<const DeviceToDeviceCopyThunk*>(thunk.get());
      auto get_argument = [&](const xla::BufferAllocation::Slice& slice) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_4(mht_4_v, 504, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");

        assert(slice.offset() == 0 && slice.size() == copy_thunk->size_bytes());
        return arguments[slice.index()];
      };
      rewriter.setInsertionPoint(op);
      rewriter.create<mlir::gpu::MemcpyOp>(
          loc, mlir::TypeRange(), mlir::ValueRange(),
          get_argument(copy_thunk->destination()),
          get_argument(copy_thunk->source()));
      continue;
    }

    auto rewrite_memset = [&](const xla::BufferAllocation::Slice& slice,
                              uint32_t memset_value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_5(mht_5_v, 520, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");

      assert(slice.offset() == 0);
      Value buffer_arg = arguments[slice.index()];
      auto element_type =
          buffer_arg.getType().cast<mlir::MemRefType>().getElementType();
      rewriter.setInsertionPoint(op);
      Value value =
          MakeBitPatternConstant(rewriter, loc, element_type, memset_value);
      rewriter.create<mlir::gpu::MemsetOp>(
          loc, mlir::TypeRange(), mlir::ValueRange(), buffer_arg, value);
    };

    if (thunk->kind() == Thunk::kMemset32BitValue) {
      const auto* memset_thunk =
          static_cast<const xla::gpu::Memset32BitValueThunk*>(thunk.get());
      rewrite_memset(memset_thunk->destination(), memset_thunk->value());
      continue;
    }
    if (thunk->kind() == Thunk::kMemzero) {
      const auto* memzero_thunk =
          static_cast<const xla::gpu::MemzeroThunk*>(thunk.get());
      rewrite_memset(memzero_thunk->destination(), 0);
      continue;
    }

    const auto* kernel_thunk = static_cast<const KernelThunk*>(thunk.get());
    rewriter.setInsertionPointToStart(gpu_module.getBody());
    SmallVector<Value, 4> kernel_args;
    for (auto kernel_arg : kernel_thunk->arguments())
      kernel_args.push_back(arguments[kernel_arg->index()]);
    auto func_type = rewriter.getType<mlir::FunctionType>(
        mlir::TypeRange(mlir::ValueRange(kernel_args)), mlir::TypeRange());
    mlir::gpu::GPUFuncOp kernel_func = rewriter.create<mlir::gpu::GPUFuncOp>(
        loc, kernel_thunk->kernel_name(), func_type);
    kernel_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                         rewriter.getUnitAttr());
    rewriter.setInsertionPointToEnd(&kernel_func.getBody().back());
    rewriter.create<mlir::gpu::ReturnOp>(loc);

    rewriter.setInsertionPoint(op);
    auto make_const_idx = [&](int64_t value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_6(mht_6_v, 563, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");

      auto attr = rewriter.getIndexAttr(value);
      return rewriter.create<mlir::arith::ConstantOp>(loc, attr).getResult();
    };
    auto make_kernel_dim3 = [&](const auto& dim3) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_7(mht_7_v, 570, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");

      return mlir::gpu::KernelDim3{make_const_idx(dim3.x),
                                   make_const_idx(dim3.y),
                                   make_const_idx(dim3.z)};
    };
    const auto& launch_dims = kernel_thunk->launch_dimensions();
    auto grid_size = make_kernel_dim3(launch_dims.block_counts());
    auto block_size = make_kernel_dim3(launch_dims.thread_counts_per_block());

    rewriter.create<mlir::gpu::LaunchFuncOp>(
        loc, kernel_func, grid_size, block_size,
        /*shared_memory_size_bytes=*/nullptr, kernel_args);
  }

  rewriter.eraseOp(op);
}

mlir::LogicalResult KernelOpsPattern::matchAndRewrite(
    mlir::ModuleOp module_op, mlir::PatternRewriter& rewriter) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_8(mht_8_v, 591, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "KernelOpsPattern::matchAndRewrite");

  SmallVector<RewriteData, 4> rewrites;

  // Get data to rewrite kernel ops without changing the IR.
  auto walk = [&](auto concrete_op) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_9(mht_9_v, 598, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "lambda");

    return module_op.walk([&](decltype(concrete_op) op) -> mlir::WalkResult {
      auto data = Match(op);
      if (!data)
        return rewriter.notifyMatchFailure(op, toString(data.takeError()));
      rewrites.emplace_back(std::move(*data));
      return mlir::success();
    });
  };
  if (walk(mlir::lmhlo::FusionOp()).wasInterrupted() ||
      walk(mlir::lmhlo::RngGetAndUpdateStateOp()).wasInterrupted() ||
      walk(mlir::lmhlo::ScatterOp()).wasInterrupted() ||
      walk(mlir::lmhlo::SelectAndScatterOp()).wasInterrupted() ||
      walk(mlir::lmhlo::SortOp()).wasInterrupted())
    return mlir::failure();

  if (rewrites.empty()) {
    return rewriter.notifyMatchFailure(module_op, "No kernel ops");
  }

  // Mark module as gpu.container_module.
  rewriter.updateRootInPlace(module_op, [&] {
    module_op->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                       rewriter.getUnitAttr());
  });

  // Replace the kernel ops with gpu.launch_func.
  mlir::SymbolTable symbol_table(module_op);
  for (const auto& data : rewrites) {
    Rewrite(data.op, rewriter, symbol_table, data.arguments, data.thunks.get(),
            data.constants, data.gpu_module_data);
  }

  return mlir::success();
}

void populateKernelOpsPattern(mlir::RewritePatternSet& patterns) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPSlmhlo_to_gpuPSkernel_ops_patternDTcc mht_10(mht_10_v, 637, "", "./tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/kernel_ops_pattern.cc", "populateKernelOpsPattern");

  patterns.add<KernelOpsPattern>(patterns.getContext());
}

}  // namespace tensorflow
