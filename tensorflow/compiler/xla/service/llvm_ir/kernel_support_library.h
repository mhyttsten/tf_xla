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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh() {
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

#include "absl/strings/string_view.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
// A thin wrapper around llvm_loop.h to make code generating structured control
// flow more readable.
class KernelSupportLibrary {
 public:
  // `b` is the llvm::IRBuilder instance used to generate LLVM IR.
  // `unroll_mode` specifies the desired LLVM unrolling behavior for every loop
  // generated by this instance of KernelSupportLibrary.
  explicit KernelSupportLibrary(
      llvm::IRBuilder<>* b,
      llvm_ir::UnrollMode unroll_mode = llvm_ir::UnrollMode::kNoUnroll,
      bool prevent_vectorization = true)
      : b_(b),
        unroll_mode_(unroll_mode),
        prevent_vectorization_(prevent_vectorization) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "KernelSupportLibrary");
}

  // Generates the following control flow structure:
  //
  //   if (`start` < `end`) {
  //     `for_body_generator(/*ind_var=*/start, /*is_first_iteration=*/true)`;
  //     for (i64 i = `start` + `step`; i s< `end`; i += `step`)
  //       `for_body_generator(/*ind_var=*/,i, /*is_first_iteration=*/false)`;
  //   }
  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<Status(llvm::Value* ind_var,
                                 bool is_first_iteration)>& for_body_generator);

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<void(llvm::Value* ind_var, bool is_first_iteration)>&
          for_body_generator) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_1(mht_1_v, 234, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    CHECK_EQ(Status::OK(),
             ForWithStatus(
                 name, start, end, step,
                 [&](llvm::Value* ind_var, bool is_first_iteration) -> Status {
                   for_body_generator(ind_var, is_first_iteration);
                   return Status::OK();
                 }));
  }

  Status ForWithStatus(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<Status(
          llvm::Value* ind_var, bool is_first_iteration)>& for_body_generator) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "ForWithStatus");

    return ForWithStatus(name, /*start=*/b_->getInt64(start),
                         /*end=*/b_->getInt64(end),
                         /*step=*/b_->getInt64(step), for_body_generator);
  }

  void For(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<void(llvm::Value* ind_var, bool is_first_iteration)>&
          for_body_generator) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_3(mht_3_v, 264, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    For(name, /*start=*/b_->getInt64(start),
        /*end=*/b_->getInt64(end),
        /*step=*/b_->getInt64(step), for_body_generator);
  }

  // Generates the following control flow structure if `peel_first_iteration` is
  // true:
  //
  //   if (`start` < `end`) {
  //     `for_body_generator(/*ind_var=*/start, /*is_first_iteration=*/,true)`;
  //     for (i64 i = `start` + `step`; i s< `end`; i += `step`)
  //       `for_body_generator(/*ind_var=*/,i, /*is_first_iteration=*/,false)`;
  //   }
  //
  // and the following if `peel_first_iteration` is false:
  //
  //   for (i64 i = `start`; i s< `end`; i += `step`)
  //     `for_body_generator(/*ind_var=*/,i,
  //                         /*is_first_iteration=*/,(i != `start`))`;
  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step, bool peel_first_iteration,
      const std::function<Status(llvm::Value* ind_var,
                                 llvm::Value* is_first_iteration)>&
          for_body_generator);

  void For(absl::string_view name, llvm::Value* start, llvm::Value* end,
           llvm::Value* step, bool peel_first_iteration,
           const std::function<void(llvm::Value* ind_var,
                                    llvm::Value* is_first_iteration)>&
               for_body_generator) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_4(mht_4_v, 299, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    TF_CHECK_OK(ForWithStatus(
        name, start, end, step, peel_first_iteration,
        [&](llvm::Value* ind_var, llvm::Value* is_first_iteration) -> Status {
          for_body_generator(ind_var, is_first_iteration);
          return Status::OK();
        }));
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step, bool peel_first_iteration,
      const std::function<Status(llvm::Value* ind_var,
                                 llvm::Value* is_first_iteration)>&
          for_body_generator) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_5(mht_5_v, 317, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "ForWithStatus");

    return ForWithStatus(
        name, /*start=*/start, /*end=*/end,
        /*step=*/llvm::ConstantInt::get(start->getType(), step),
        peel_first_iteration, for_body_generator);
  }

  void For(absl::string_view name, llvm::Value* start, llvm::Value* end,
           int64_t step, bool peel_first_iteration,
           const std::function<void(llvm::Value* ind_var,
                                    llvm::Value* is_first_iteration)>&
               for_body_generator) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_6(mht_6_v, 332, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    For(name, /*start=*/start, /*end=*/end,
        /*step=*/llvm::ConstantInt::get(start->getType(), step),
        peel_first_iteration, for_body_generator);
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_7(mht_7_v, 345, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "ForWithStatus");

    return ForWithStatus(name, start, end, step,
                         /*peel_first_iteration=*/false,
                         [&](llvm::Value* indvar, llvm::Value*) -> Status {
                           return for_body_generator(indvar);
                         });
  }

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_8(mht_8_v, 360, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    For(name, start, end, step,
        /*peel_first_iteration=*/false, [&](llvm::Value* indvar, llvm::Value*) {
          return for_body_generator(indvar);
        });
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_9(mht_9_v, 374, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "ForWithStatus");

    return ForWithStatus(name, start, end,
                         llvm::ConstantInt::get(start->getType(), step),
                         /*peel_first_iteration=*/false,
                         [&](llvm::Value* indvar, llvm::Value*) -> Status {
                           return for_body_generator(indvar);
                         });
  }

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_10(mht_10_v, 390, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    For(name, start, end, llvm::ConstantInt::get(start->getType(), step),
        for_body_generator);
  }

  Status ForWithStatus(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_11(mht_11_v, 401, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "ForWithStatus");

    return ForWithStatus(name, /*start=*/b_->getInt64(start),
                         /*end=*/b_->getInt64(end),
                         /*step=*/b_->getInt64(step), for_body_generator);
  }

  void For(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_12(mht_12_v, 413, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "For");

    For(name, /*start=*/b_->getInt64(start),
        /*end=*/b_->getInt64(end),
        /*step=*/b_->getInt64(step), for_body_generator);
  }

  // Generates the following control flow structure:
  //
  //   if (`condition`)
  //     `true_block_generator()`;
  //   else
  //      `false_block_generator()`;
  // The else is skipped if false_block_generator is null.
  Status IfWithStatus(
      absl::string_view name, llvm::Value* condition,
      const std::function<Status()>& true_block_generator,
      const std::function<Status()>& false_block_generator = nullptr);

  Status IfWithStatus(
      llvm::Value* condition,
      const std::function<Status()>& true_block_generator,
      const std::function<Status()>& false_block_generator = []() -> Status {
        return Status::OK();
      }) {
    return IfWithStatus("", condition, true_block_generator,
                        false_block_generator);
  }

  void If(llvm::Value* condition,
          const std::function<void()>& true_block_generator,
          const std::function<void()>& false_block_generator = nullptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_13(mht_13_v, 446, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "If");

    If("", condition, true_block_generator, false_block_generator);
  }

  void If(absl::string_view name, llvm::Value* condition,
          const std::function<void()>& true_block_generator,
          const std::function<void()>& false_block_generator = nullptr) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_14(mht_14_v, 456, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "If");

    if (false_block_generator != nullptr) {
      TF_CHECK_OK(IfWithStatus(
          name, condition,
          [&]() {
            true_block_generator();
            return Status::OK();
          },
          [&]() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_15(mht_15_v, 467, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "lambda");

            false_block_generator();
            return Status::OK();
          }));
    } else {
      TF_CHECK_OK(IfWithStatus(name, condition, [&]() {
        true_block_generator();
        return Status::OK();
      }));
    }
  }

  using ArgumentVector = absl::Span<llvm::Value* const>;

  // Generates the following control flow structure:
  //
  //  define @`kernel_name`(arg0, arg1, ... arg`arguments.size()`) {
  //    kernel_body_generator({arg0, arg1, ... arg`arguments.size()`});
  //  }
  //
  //  ...
  //  call @`kernel_name`(arguments[0], arguments[1] ...)
  //  ...
  //
  // If a function called `kernel_name` is already present in the module then
  // that function is re-used.  In that sense we're using the llvm::Module as a
  // cache of outlined kernels, keyed by function name.
  //
  // If any of the values in `arguments` is nullptr (i.e. a nullptr
  // llvm::Value*) then we ignore it when generating LLVM IR, and instead pass
  // in a nullptr llvm::Value* in its position to `kernel_body_generator`.
  // Currently we only support at most one nullptr value in `arguments`.
  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, ArgumentVector arguments,
      const std::function<void(ArgumentVector)>& kernel_body_generator);

  // Thin wrappers around the more general EmitAndCallOutlinedKernel above.
  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, llvm::Value* arg0, llvm::Value* arg1,
      llvm::Value* arg2,
      const std::function<void(llvm::Value*, llvm::Value*, llvm::Value*)>&
          kernel_body_generator) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_16(mht_16_v, 514, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "EmitAndCallOutlinedKernel");

    EmitAndCallOutlinedKernel(module_config, b, kernel_name, {arg0, arg1, arg2},
                              [&](ArgumentVector args) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_17(mht_17_v, 519, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "lambda");

                                kernel_body_generator(args[0], args[1],
                                                      args[2]);
                              });
  }

  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, llvm::Value* arg0, llvm::Value* arg1,
      llvm::Value* arg2, llvm::Value* arg3,
      const std::function<void(llvm::Value*, llvm::Value*, llvm::Value*,
                               llvm::Value*)>& kernel_body_generator) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_18(mht_18_v, 534, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "EmitAndCallOutlinedKernel");

    EmitAndCallOutlinedKernel(
        module_config, b, kernel_name, {arg0, arg1, arg2, arg3},
        [&](ArgumentVector args) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTh mht_19(mht_19_v, 540, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h", "lambda");

          kernel_body_generator(args[0], args[1], args[2], args[3]);
        });
  }

 private:
  llvm::IRBuilder<>* b_;
  llvm_ir::UnrollMode unroll_mode_;
  bool prevent_vectorization_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_
