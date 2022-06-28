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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/ir_function.h"

#include <iterator>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace cpu {

static std::vector<llvm::Type*> GetComputeFunctionParams(
    llvm::Module* llvm_module, const int64_t num_dynamic_loop_bounds) {
  llvm::Type* i8_ptr_type = llvm::Type::getInt8PtrTy(llvm_module->getContext());
  llvm::Type* i8_ptr_ptr_type = i8_ptr_type->getPointerTo();
  llvm::Type* i64_ptr_type =
      llvm::Type::getInt64PtrTy(llvm_module->getContext());
  std::vector<llvm::Type*> compute_function_params(
      {i8_ptr_type, i8_ptr_type, i8_ptr_ptr_type, i8_ptr_ptr_type,
       i8_ptr_type});
  if (num_dynamic_loop_bounds > 0) {
    compute_function_params.push_back(i64_ptr_type);
  }
  compute_function_params.push_back(i64_ptr_type);
  return compute_function_params;
}

IrFunction::IrFunction(const std::string& function_name,
                       llvm::Function::LinkageTypes linkage,
                       const HloModuleConfig& module_config,
                       llvm::Module* llvm_module, llvm::IRBuilder<>* b,
                       int64_t num_dynamic_loop_bounds)
    : b_(b),
      llvm_module_(llvm_module),
      caller_insert_point_guard_(*b),
      num_dynamic_loop_bounds_(num_dynamic_loop_bounds) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_0(mht_0_v, 223, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "IrFunction::IrFunction");

  Initialize(function_name, linkage, module_config);
}

IrFunction::~IrFunction() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_1(mht_1_v, 230, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "IrFunction::~IrFunction");

  // Branch to function return.
  b_->CreateBr(return_block_);
}

DynamicLoopBounds IrFunction::GetDynamicLoopBounds() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "IrFunction::GetDynamicLoopBounds");

  DynamicLoopBounds dynamic_loop_bounds(num_dynamic_loop_bounds_);
  for (int i = 0; i < num_dynamic_loop_bounds_; ++i) {
    dynamic_loop_bounds[i].first = GetDynamicLoopBound(i * 2 + 0);
    dynamic_loop_bounds[i].second = GetDynamicLoopBound(i * 2 + 1);
  }
  return dynamic_loop_bounds;
}

void IrFunction::Initialize(const std::string& function_name,
                            llvm::Function::LinkageTypes linkage,
                            const HloModuleConfig& module_config) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "IrFunction::Initialize");

  // The function signature is:
  //   void function(i8* retval, i8* run_options, i8** params, i8**
  //   buffer_table,
  //                 i64* dynamic_loop_bounds, i64* prof_counters)
  //
  // For thread local functions:
  //   retval: points to the returned value.
  //   params: address of an array with pointers to parameters.
  //   buffer_table: is null
  //
  // For global functions:
  //   retval: is null
  //   params: is null
  //   buffer_table: address of an array with pointers to temporary buffers and
  //     entry computation parameters (but not to constant buffers).
  //
  // Therefore, the generated function's signature (FunctionType) is statically
  // determined - parameter unpacking is done in code generated into the
  // function, rather than by a prologue dictated by the platform ABI.
  //
  //                      /--------------\
  //   retval ----------> | return value |
  //                      \--------------/
  //
  //                      /-------------------------------\
  //   run_options -----> | xla::ExecutableRunOptions |
  //                      \-------------------------------/
  //
  //                     /---------------------------------------------\
  //   params -------->  |  param 0  |  param 1  | ..... |  param N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | param 0 |  | param 1 |         | param N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                     /---------------------------------------------\
  //   buffer_table--->  |  buff  0  |  guff  1  | ..... |  buff  N-1  |
  //                     |   addr    |   addr    |       |   addr      |
  //                     \---------------------------------------------/
  //                          |           |                   |
  //                          |           |                   |
  //                          V           V                   V
  //                     /---------\  /---------\         /-----------\
  //                     | temp  0 |  | temp  1 |         | temp  N-1 |
  //                     \---------/  \---------/         \-----------/
  //
  //                        /--------------------------------------------\
  // dynamic loop bounds -> | outer_dim0_start | outer_dim0_limit | .....|
  //  (elided for aot)      \--------------------------------------------/
  //
  //                     /---------------------------------------------\
  //   prof counters ->  | counter 0 | counter 1 | ..... | counter N-1 |
  //                     \---------------------------------------------/

  // Even though the type of params and buffer_table is void** in the host's
  // view, in LLVM IR this is represented by i8*, similarly to void*. It's up to
  // the code to use GEPs to unravel the indirection layers.
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(llvm_module_->getContext()),
      /*Params=*/
      GetComputeFunctionParams(llvm_module_, num_dynamic_loop_bounds_),
      /*isVarArg=*/false);

  // Functions with local linkage get an inlining bonus.  Because we know
  // a-priori that embedded functions (non-entry functions) will not have its
  // name resolved, give it local linkage.
  function_ = llvm_ir::CreateCpuFunction(function_type, linkage, module_config,
                                         function_name, llvm_module_);

  // Set meaningful names for the function's arguments: useful for debugging.
  llvm::Function::arg_iterator arg_iter = function_->arg_begin();
  arg_iter->setName("retval");
  result_arg_ = &*arg_iter;
  (++arg_iter)->setName("run_options");
  exec_run_options_arg_ = &*arg_iter;
  (++arg_iter)->setName("params");
  parameters_arg_ = &*arg_iter;
  (++arg_iter)->setName("buffer_table");
  buffer_table_arg_ = &*arg_iter;
  (++arg_iter)->setName("status");
  status_arg_ = &*arg_iter;
  if (num_dynamic_loop_bounds_ > 0) {
    (++arg_iter)->setName("dynamic_loop_bounds");
    dynamic_loop_bounds_arg_ = &*arg_iter;
  }
  (++arg_iter)->setName("prof_counters");
  profile_counters_arg_ = &*arg_iter;

  // We know a-priori that the function arguments are guaranteed to point to
  // disjoint objects.
  llvm::Argument* retval = result_arg();
  for (llvm::Argument& argument : function_->args()) {
    // However, the return buffer aliases the temporaries and thus cannot be
    // marked noalias.
    if (&argument == retval) {
      continue;
    }
    function_->addParamAttr(argument.getArgNo(), llvm::Attribute::NoAlias);
  }

  return_block_ =
      llvm::BasicBlock::Create(/*Context=*/llvm_module_->getContext(),
                               /*Name=*/"return", /*Parent=*/function_);

  b_->SetInsertPoint(return_block_);
  b_->CreateRetVoid();

  b_->SetInsertPoint(llvm::BasicBlock::Create(
      /*Context=*/llvm_module_->getContext(),
      /*Name=*/"entry",
      /*Parent=*/function_,
      /*InsertBefore=*/return_block_));
}

llvm::Value* IrFunction::GetDynamicLoopBound(const int64_t offset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_4(mht_4_v, 375, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "IrFunction::GetDynamicLoopBound");

  CHECK_GT(num_dynamic_loop_bounds_, 0);
  CHECK_LT(offset, num_dynamic_loop_bounds_ * 2);
  llvm::Type* type =
      dynamic_loop_bounds_arg_->getType()->getPointerElementType();
  auto gep = b_->CreateGEP(type, CHECK_NOTNULL(dynamic_loop_bounds_arg_),
                           b_->getInt64(offset));
  return b_->CreateLoad(type, gep, "dynamic_loop_bound_" + llvm::Twine(offset));
}

llvm::Value* EncodeArrayFunctionArguments(
    absl::Span<llvm::Value* const> arguments, absl::string_view name,
    llvm::IRBuilder<>* b) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_5(mht_5_v, 391, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "EncodeArrayFunctionArguments");

  llvm::Value* arguments_buffer;
  llvm::Type* int8ptr_ty = b->getInt8PtrTy();
  if (arguments.empty()) {
    arguments_buffer = llvm::Constant::getNullValue(int8ptr_ty->getPointerTo());
  } else {
    arguments_buffer = llvm_ir::EmitAllocaAtFunctionEntryWithCount(
        int8ptr_ty, b->getInt32(arguments.size()),
        absl::StrCat(name, "_parameter_addresses"), b);

    for (size_t i = 0; i < arguments.size(); i++) {
      llvm::Value* parameter_as_i8ptr = b->CreateBitCast(
          arguments[i], b->getInt8PtrTy(),
          absl::StrCat(name, "_parameter_", i, "_address_as_i8ptr"));
      llvm::Value* slot_in_param_addresses =
          b->CreateInBoundsGEP(int8ptr_ty, arguments_buffer, b->getInt64(i));
      b->CreateStore(parameter_as_i8ptr, slot_in_param_addresses);
    }
  }
  return arguments_buffer;
}

// Emits code to allocate an array of parameter address pointers, and store
// each address from 'parameter_addresses'.
// Returns an array of compute function call arguments (including parameter
// address buffer).
std::vector<llvm::Value*> GetArrayFunctionCallArguments(
    absl::Span<llvm::Value* const> parameter_addresses, llvm::IRBuilder<>* b,
    absl::string_view name, llvm::Value* return_value_buffer,
    llvm::Value* exec_run_options_arg, llvm::Value* buffer_table_arg,
    llvm::Value* status_arg, llvm::Value* profile_counters_arg) {
  llvm::Value* parameter_addresses_buffer =
      EncodeArrayFunctionArguments(parameter_addresses, name, b);

  const auto to_int8_ptr = [=](llvm::Value* ptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_6(mht_6_v, 428, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "lambda");

    return b->CreatePointerCast(ptr, b->getInt8PtrTy());
  };
  return std::vector<llvm::Value*>{to_int8_ptr(return_value_buffer),
                                   to_int8_ptr(exec_run_options_arg),
                                   parameter_addresses_buffer,
                                   buffer_table_arg,
                                   status_arg,
                                   profile_counters_arg};
}

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
Status EmitCallToParallelForkJoin(
    const std::vector<llvm::Value*>& arguments, const Shape& shape,
    const std::vector<int64_t>& dimension_partition_counts,
    llvm::IRBuilder<>* b, llvm::Function* parallel_function,
    const std::string& name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSir_functionDTcc mht_7(mht_7_v, 449, "", "./tensorflow/compiler/xla/service/cpu/ir_function.cc", "EmitCallToParallelForkJoin");

  llvm::Module* module = b->GetInsertBlock()->getModule();

  // Build ParallelForkJoin function type.
  std::vector<llvm::Type*> compute_function_params =
      GetComputeFunctionParams(module, /*num_dynamic_loop_bounds=*/0);
  // Number of parallel compute functions.
  compute_function_params.push_back(b->getInt32Ty());
  // Array of partitions. There is an array element for each
  // partition x partition_dim x 2 (for dimension start and limit).
  compute_function_params.push_back(
      llvm::Type::getInt64PtrTy(module->getContext()));
  // Number of partitioned most-major dimensions in 'shape'.
  compute_function_params.push_back(b->getInt32Ty());
  // Function pointer for compute function to be dispatched in parallel.
  compute_function_params.push_back(
      llvm::Type::getInt8PtrTy(module->getContext()));

  llvm::FunctionType* fork_join_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(module->getContext()),
      /*Params=*/compute_function_params,
      /*isVarArg=*/false);

  llvm::Function* fork_join_func = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(runtime::kParallelForkJoinSymbolName,
                                fork_join_type)
          .getCallee());
  fork_join_func->setCallingConv(llvm::CallingConv::C);
  fork_join_func->setDoesNotThrow();

  // Add common compute function arguments.
  std::vector<llvm::Value*> fork_join_arguments(arguments);

  // Create ShapePartitionIterator to generate all partitions of 'shape'.
  ShapePartitionIterator partition_iterator(shape, dimension_partition_counts);
  const int64_t num_partitions = partition_iterator.GetTotalPartitionCount();
  // Add argument specifying the number of parallel partitions.
  fork_join_arguments.push_back(b->getInt32(num_partitions));

  // The number of partitioned most-major dimensions in 'shape'.
  const int32_t num_partitioned_dims = dimension_partition_counts.size();
  // A dimension partition consists of two elements: [start_index, limit_index).
  const int32_t dim_partition_size = 2;
  // Calculate array partition stride.
  const int32_t array_partition_stride =
      num_partitioned_dims * dim_partition_size;
  // Calculate the total number of elements in the partition array.
  const int32_t partition_array_size =
      dim_partition_size * num_partitioned_dims * num_partitions;

  // Store dimension partition values as llvm constants in 'partitions'.
  // See comments in runtime_fork_join.cc for array layout description.
  std::vector<llvm::Constant*> partitions(partition_array_size);
  for (int32_t i = 0; i < num_partitions; ++i) {
    std::vector<std::pair<int64_t, int64_t>> dim_partitions =
        partition_iterator.GetPartition(i);
    CHECK_EQ(num_partitioned_dims, dim_partitions.size());
    const int32_t partition_index = i * array_partition_stride;
    for (int32_t j = 0; j < num_partitioned_dims; ++j) {
      const std::pair<int64_t, int64_t>& dim_partition = dim_partitions[j];
      const int32_t index = partition_index + j * dim_partition_size;
      // Store partition [dim_start, dim_limit) intervals for each dimension.
      partitions[index] = b->getInt64(dim_partition.first);
      partitions[index + 1] =
          b->getInt64(dim_partition.first + dim_partition.second);
    }
  }

  // Create global variable out of dimension partitions in 'partitions'.
  llvm::ArrayType* partitions_array_type =
      llvm::ArrayType::get(b->getInt64Ty(), partition_array_size);
  llvm::Constant* partitions_array =
      llvm::ConstantArray::get(partitions_array_type, partitions);
  llvm::GlobalVariable* global_partitions_array = new llvm::GlobalVariable(
      /*M=*/*module,
      /*Ty=*/partitions_array_type,
      /*isConstant=*/true,
      /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
      /*Initializer=*/partitions_array,
      /*Name=*/
      absl::StrCat(name, "_parallel_dimension_partitions"));

  // Add argument specifying parallel dimension partitions.
  fork_join_arguments.push_back(
      b->CreateBitCast(global_partitions_array,
                       llvm::Type::getInt64PtrTy(module->getContext())));
  // Add argument specifying the number of partitioned most-major dimensions.
  fork_join_arguments.push_back(b->getInt32(num_partitioned_dims));
  // Add argument for parallel compute function pointer.
  fork_join_arguments.push_back(
      b->CreateBitCast(parallel_function, b->getInt8PtrTy()));
  // Emit call to parallel fork/join.
  b->CreateCall(fork_join_func, fork_join_arguments);

  return Status::OK();
}

}  // namespace cpu
}  // namespace xla
