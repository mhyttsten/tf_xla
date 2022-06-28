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

// This is the side effect definition file for TensorFlow.
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh() {
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


#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace TF {
namespace ResourceEffects {

struct Variable : ::mlir::SideEffects::Resource::Base<Variable> {
  StringRef getName() final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_0(mht_0_v, 196, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "Variable"; }
};

struct Stack : ::mlir::SideEffects::Resource::Base<Stack> {
  StringRef getName() final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_1(mht_1_v, 203, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "Stack"; }
};

struct TensorArray : ::mlir::SideEffects::Resource::Base<TensorArray> {
  StringRef getName() final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_2(mht_2_v, 210, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "TensorArray"; }
};

struct Summary : ::mlir::SideEffects::Resource::Base<Summary> {
  StringRef getName() final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_3(mht_3_v, 217, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "Summary"; }
};

struct LookupTable : ::mlir::SideEffects::Resource::Base<LookupTable> {
  StringRef getName() final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_4(mht_4_v, 224, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "LookupTable"; }
};

struct DatasetSeedGenerator
    : ::mlir::SideEffects::Resource::Base<DatasetSeedGenerator> {
  StringRef getName() final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_5(mht_5_v, 232, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "DatasetSeedGenerator"; }
};

struct DatasetMemoryCache
    : ::mlir::SideEffects::Resource::Base<DatasetMemoryCache> {
  StringRef getName() final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_6(mht_6_v, 240, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "DatasetMemoryCache"; }
};

struct DatasetIterator : ::mlir::SideEffects::Resource::Base<DatasetIterator> {
  StringRef getName() final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_7(mht_7_v, 247, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "DatasetIterator"; }
};

// Special resource type to track TPU Embedding specific ops, which must execute
// but do not have side effects with one another or with resource variable ops.
struct TPUEmbedding : ::mlir::SideEffects::Resource::Base<TPUEmbedding> {
  StringRef getName() final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_8(mht_8_v, 256, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "TPUEmbedding"; }
};

// Resource corresponding to GeneratorOp.
struct GeneratorOp : public ::mlir::SideEffects::Resource::Base<GeneratorOp> {
  StringRef getName() final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_9(mht_9_v, 264, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<Default Generator>"; }
};

struct Send : public ::mlir::SideEffects::Resource::Base<Send> {
  StringRef getName() final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_10(mht_10_v, 271, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<Send>"; }
};

struct Recv : public ::mlir::SideEffects::Resource::Base<Recv> {
  StringRef getName() final {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_11(mht_11_v, 278, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<Recv>"; }
};

struct RandomGenerator
    : public ::mlir::SideEffects::Resource::Base<RandomGenerator> {
  StringRef getName() final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_12(mht_12_v, 286, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<RandomGenerator>"; }
};

struct TPUExecute : public ::mlir::SideEffects::Resource::Base<TPUExecute> {
  StringRef getName() final {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_13(mht_13_v, 293, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<TPUExecute>"; }
};

struct MustExecute : public ::mlir::SideEffects::Resource::Base<MustExecute> {
  StringRef getName() final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_side_effectsDTh mht_14(mht_14_v, 300, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h", "getName");
 return "<MustExecute>"; }
};

}  // namespace ResourceEffects
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
