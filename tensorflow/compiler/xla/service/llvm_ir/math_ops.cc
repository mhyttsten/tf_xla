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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSmath_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSmath_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSmath_opsDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/math_ops.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace llvm_ir {

llvm::Value* EmitFastTanh(llvm::IRBuilder<>* b, llvm::Value* input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSmath_opsDTcc mht_0(mht_0_v, 191, "", "./tensorflow/compiler/xla/service/llvm_ir/math_ops.cc", "EmitFastTanh");

  llvm::Type* type = input->getType();

  // For small values of x, we can approximate tanh(x)=x. For extremely small
  // values of x (|x| < 1e-37), the other approximation evaluates tanh(x) = 0.
  const auto kCanUseApprox = 0.0004;
  auto abs_x =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b);
  auto use_aprox =
      b->CreateFCmpOLT(abs_x, llvm::ConstantFP::get(type, kCanUseApprox));

  // Clamp the input to [-9, 9].
  //
  // To simplify the code base until it's an issue, don't have a slow min/max in
  // this approximation.
  llvm::Value* input_clamped = llvm_ir::EmitFloatMin(
      llvm_ir::EmitFloatMax(input, llvm::ConstantFP::get(type, -9.0), b,
                            /*enable_fast_min_max=*/true),
      llvm::ConstantFP::get(type, 9.0), b, /*enable_fast_min_max=*/true);

  static constexpr std::array<float, 7> numerator_coeffs{
      -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
      5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
      4.89352455891786e-03f};

  static constexpr std::array<float, 4> denominator_coeffs{
      1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
      4.89352518554385e-03f};

  llvm::Value* input_squared = b->CreateFMul(input_clamped, input_clamped);
  llvm::Value* numerator = llvm::ConstantFP::get(type, numerator_coeffs[0]);
  for (int i = 1; i < numerator_coeffs.size(); i++) {
    numerator = b->CreateFAdd(b->CreateFMul(input_squared, numerator),
                              llvm::ConstantFP::get(type, numerator_coeffs[i]));
  }

  numerator = b->CreateFMul(input_clamped, numerator);

  llvm::Value* denominator = llvm::ConstantFP::get(type, denominator_coeffs[0]);
  for (int i = 1; i < denominator_coeffs.size(); i++) {
    denominator =
        b->CreateFAdd(b->CreateFMul(input_squared, denominator),
                      llvm::ConstantFP::get(type, denominator_coeffs[i]));
  }

  return b->CreateSelect(use_aprox, input,
                         b->CreateFDiv(numerator, denominator));
}

}  // namespace llvm_ir
}  // namespace xla
