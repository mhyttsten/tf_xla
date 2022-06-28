/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/primitive_util.h"

namespace xla {
namespace cpu {

// Abstract interface for classes providing information about the target we're
// compiling for.
class TargetMachineFeatures {
 public:
  static constexpr int kX86AvxVectorByteSize = 32;

  // Input and output tensor buffers must be aligned to this many bytes if we
  // want to call an Eigen backed GEMM or Convolution.
  static constexpr int kEigenExpectedTensorAlignment = 16;

  // Return the vectorization factor, which is the number of bytes of data
  // explicitly vectorized routines will try to process at once.
  virtual int vectorization_factor_in_bytes() const = 0;

  // Return the size of the largest vector size in bytes.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  virtual int vector_register_byte_size(
      const llvm::Function& function) const = 0;

  // Return the number of elements of type `type` that can fit into the largest
  // vector register available.  We need to pass in "function" since llvm
  // functions can contain annotations for specializing them to specific
  // micro-architectures (though currently XLA does not use this functionality).
  virtual int vector_register_num_elements(const llvm::Function& function,
                                           PrimitiveType type) const = 0;

  // Return the number of vector registers.  We need to pass in
  // "function" since llvm functions can contain annotations for specializing
  // them to specific micro-architectures (though currently XLA does not use
  // this functionality).
  virtual int vector_register_count(const llvm::Function& function) const = 0;

  // Returns the minimum alignment for a buffer of size size_bytes.
  virtual int64_t minimum_alignment_for_allocation(
      int64_t size_bytes) const = 0;

  virtual ~TargetMachineFeatures() = default;
};

// Implements the TargetMachineFeatures interface using an llvm::TargetMachine.
class LLVMTargetMachineFeatures : public TargetMachineFeatures {
 public:
  static constexpr int kX86AvxVectorByteSize = 32;

  LLVMTargetMachineFeatures(llvm::TargetMachine* target_machine)
      : target_machine_(target_machine) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh mht_0(mht_0_v, 243, "", "./tensorflow/compiler/xla/service/cpu/target_machine_features.h", "LLVMTargetMachineFeatures");
}

  int vectorization_factor_in_bytes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh mht_1(mht_1_v, 248, "", "./tensorflow/compiler/xla/service/cpu/target_machine_features.h", "vectorization_factor_in_bytes");

    // Ideally this should be a function of the cache line size (which we can
    // get from llvm::TargetTransformInfo::getCacheLineSize) of the target
    // machine.  Guess a value of 128 bytes for now.
    return 128;
  }

  int vector_register_byte_size(const llvm::Function& function) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/service/cpu/target_machine_features.h", "vector_register_byte_size");

    llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(function);
    return tti->getRegisterBitWidth(
               llvm::TargetTransformInfo::RGK_FixedWidthVector) /
           8;
  }

  int vector_register_num_elements(const llvm::Function& function,
                                   PrimitiveType type) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh mht_3(mht_3_v, 269, "", "./tensorflow/compiler/xla/service/cpu/target_machine_features.h", "vector_register_num_elements");

    return vector_register_byte_size(function) /
           (primitive_util::BitWidth(type) / 8);
  }

  int vector_register_count(const llvm::Function& function) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPStarget_machine_featuresDTh mht_4(mht_4_v, 277, "", "./tensorflow/compiler/xla/service/cpu/target_machine_features.h", "vector_register_count");

    llvm::TargetTransformInfo* tti = GetTargetTransformInfoFor(function);
    return static_cast<int>(tti->getNumberOfRegisters(
        tti->getRegisterClassForType(/*Vector=*/true)));
  }

  int64_t minimum_alignment_for_allocation(int64_t size_bytes) const override;

 private:
  llvm::TargetTransformInfo* GetTargetTransformInfoFor(
      const llvm::Function& function) const;

  // This cache saves us from having to create a llvm::TargetTransformInfo for
  // every call to GetTargetTransformInfoFor (creating a TargetTransformInfo
  // costs one heap allocation on X86).
  //
  // This is mutated from within `GetTargetTransformInfoFor` which is
  // semantically a getter (and thus `const`); and is therefore declared
  // mutable.  Making this mutable is okay because it has cache semantics.
  mutable absl::flat_hash_map<const llvm::Function*, llvm::TargetTransformInfo>
      target_transform_info_cache_;
  llvm::TargetMachine* target_machine_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TARGET_MACHINE_FEATURES_H_
