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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"

namespace mlir {
namespace TF {

SmallVector<int64_t, 4> ReversePermutation(ArrayRef<int64_t> permutation) {
  SmallVector<int64_t, 4> reverse(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    reverse[permutation[i]] = i;
  }
  return reverse;
}

SmallVector<int64_t, 4> GetDataFormatPermutation(StringRef from, StringRef to) {
  if (from == "NHWC" && to == "NCHW") {
    return {0, 3, 1, 2};
  } else if (from == "NCHW" && to == "NHWC") {
    return {0, 2, 3, 1};
  } else {
    return {};
  }
}

// Shuffle elements in the `attr` according to the permutation. Optional
// `inner_size` allows to shuffle array attributes created from rank 2 tensors
// on outer dimension only.
ArrayAttr ShuffleArrayAttr(ArrayAttr attr, ArrayRef<int64_t> permutation,
                           int inner_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.cc", "ShuffleArrayAttr");

  if (attr.empty()) return attr;

  assert(attr.size() % inner_size == 0);
  assert(attr.size() / inner_size == permutation.size());

  SmallVector<Attribute, 8> values{attr.begin(), attr.end()};
  SmallVector<Attribute, 8> shuffled(values.size());

  for (size_t i = 0; i < permutation.size(); ++i) {
    for (size_t j = 0; j < inner_size; ++j) {
      shuffled[i * inner_size + j] = values[permutation[i] * inner_size + j];
    }
  }

  return ArrayAttr::get(attr.getContext(), shuffled);
}

// Shuffle ranked tensor dimensions according to the permutation.
Type ShuffleRankedTensorType(Type type, ArrayRef<int64_t> permutation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc mht_1(mht_1_v, 234, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.cc", "ShuffleRankedTensorType");

  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    ArrayRef<int64_t> shape = ranked_type.getShape();
    assert(permutation.size() == shape.size());

    SmallVector<int64_t, 4> new_shape(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i)
      new_shape[i] = shape[permutation[i]];

    return RankedTensorType::get(new_shape, ranked_type.getElementType());
  }

  return type;
}

bool AreCancellablePermutations(DenseIntElementsAttr perm0,
                                DenseIntElementsAttr perm1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_ops_layout_helperDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.cc", "AreCancellablePermutations");

  if (perm0.getNumElements() == 0 || perm1.getNumElements() == 0) return false;
  if (perm0.getNumElements() != perm1.getNumElements()) return false;

  SmallVector<int64_t, 8> perm0_values;
  for (const auto &value : perm0.getValues<APInt>())
    perm0_values.push_back(value.getSExtValue());

  SmallVector<int64_t, 8> perm1_values;
  for (const auto &value : perm1.getValues<APInt>())
    perm1_values.push_back(value.getSExtValue());

  for (int i = 0; i < perm0_values.size(); ++i) {
    if (perm0_values[perm1_values[i]] != i) return false;
  }

  return true;
}

}  // namespace TF
}  // namespace mlir
