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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc() {
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

#include "tensorflow/compiler/mlir/tfr/utils/utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"

namespace mlir {
namespace TFR {
namespace {

// TODO(b/174692018): Use the official allowlist of the unregistered attrs.
const llvm::StringSet<>& GetAllowedAttributes() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "GetAllowedAttributes");

  static auto* const ops = new llvm::StringSet<>({"device", "_tpu_replicate"});
  return *ops;
}

// Some TFL optional attributes may not appear in their corresponding TF op
// attributes.
const llvm::StringSet<>& GetOptionalAttributes() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "GetOptionalAttributes");

  static auto* const ops =
      new llvm::StringSet<>({"asymmetric_quantize_inputs"});
  return *ops;
}

void CollectAllowedAttrs(CallOp src, NamedAttrList* attrs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_2(mht_2_v, 217, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "CollectAllowedAttrs");

  for (auto& attr : src->getAttrs()) {
    if (GetAllowedAttributes().contains(attr.getName().strref())) {
      attrs->append(attr);
    }
  }
}

// Adds `attrs` to all the operations between `begin` and `end` in the same
// block. Does not include `end`.
void AddAttributesInSameBlock(Block::iterator begin, Block::iterator end,
                              const NamedAttrList& attrs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_3(mht_3_v, 231, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "AddAttributesInSameBlock");

  for (Block::iterator it = begin; it != end; ++it) {
    for (auto& attr : attrs) {
      it->setAttr(attr.getName(), attr.getValue());
    }
  }
}

// Adds `attrs` to all the operations between `begin` and `end`. Does not
// include `end`. The operations might be across multiple  blocks.
void AddAttributes(Block::iterator begin, Block::iterator end,
                   const NamedAttrList& attrs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "AddAttributes");

  if (begin->getBlock() == end->getBlock()) {
    AddAttributesInSameBlock(begin, end, attrs);
  } else {
    Region::iterator begin_block = Region::iterator(begin->getBlock());
    Region::iterator end_block = Region::iterator(end->getBlock());
    AddAttributesInSameBlock(begin, begin_block->end(), attrs);
    for (Region::iterator it = ++begin_block; it != end_block; ++it) {
      AddAttributesInSameBlock(it->begin(), it->end(), attrs);
    }
  }
}

}  // namespace

std::string GetComposeFuncName(StringRef tf_op_name) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_5(mht_5_v, 263, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "GetComposeFuncName");

  std::string compose_func_name;
  for (int i = 0; i < tf_op_name.size(); ++i) {
    if (tf_op_name[i] == '_') {
      // The field name must not contain "_"s. "_Arg" and "_RetVal" are special
      // op names and we can return empty string to skip the decomposition.
      return {};
    }
    if (tf_op_name[i] == '.') {
      compose_func_name.push_back('_');
    } else if (tf_op_name[i] >= 'A' && tf_op_name[i] <= 'Z') {
      compose_func_name.push_back('_');
      compose_func_name.push_back(tf_op_name[i] + 'a' - 'A');
    } else {
      compose_func_name.push_back(tf_op_name[i]);
    }
  }
  return compose_func_name;
}

std::string GetTFOpName(StringRef compose_func_name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_6(mht_6_v, 286, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "GetTFOpName");

  std::string tf_op_name;
  bool after_underscore = false;
  for (int i = 0; i < compose_func_name.size(); ++i) {
    if (compose_func_name[i] >= 'A' && compose_func_name[i] <= 'Z') {
      // The field name must not contain uppercase letters.
      return {};
    }
    if (after_underscore) {
      if (compose_func_name[i] >= 'a' && compose_func_name[i] <= 'z') {
        tf_op_name.push_back(compose_func_name[i] + 'A' - 'a');
        after_underscore = false;
      } else {
        // The character after a "_" must be a lowercase letter.
        return {};
      }
    } else if (compose_func_name[i] == '_') {  // first time visit '_'
      if (i + 1 < compose_func_name.size() && compose_func_name[i + 1] == '_') {
        tf_op_name.push_back('.');
        i++;
      }
      after_underscore = true;
    } else {
      tf_op_name.push_back(compose_func_name[i]);
    }
  }
  if (after_underscore) {
    // Trailing "_".
    return {};
  }
  return tf_op_name;
}

LogicalResult ValidateAttrs(Operation* src, const StringSet<>& registered) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_7(mht_7_v, 322, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "ValidateAttrs");

  for (auto& attr : src->getAttrs()) {
    StringRef attr_name = attr.getName().strref();

    if (!registered.contains(attr_name) &&
        !(GetAllowedAttributes().contains(attr_name) ||
          GetOptionalAttributes().contains(attr_name))) {
      src->emitError("Denied unregistered attribute was found: " + attr_name);
      return failure();
    }
  }
  return success();
}

LogicalResult CopyAllowedUnregisteredAttrs(Operation* src, CallOp dst,
                                           const StringSet<>& registered) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_8(mht_8_v, 340, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "CopyAllowedUnregisteredAttrs");

  for (auto& attr : src->getAttrs()) {
    StringRef attr_name = attr.getName().strref();
    // Skip the registered or optional attribute.
    if (registered.contains(attr_name) ||
        GetOptionalAttributes().contains(attr_name))
      continue;

    // Unregistered attribute.
    if (GetAllowedAttributes().contains(attr_name)) {
      dst->setAttr(attr.getName(), attr.getValue());
    } else {
      src->emitError("Denied unregistered attribute was found: " + attr_name);
      return failure();
    }
  }
  return success();
}

LogicalResult CopyNonSymbolRefAttrs(CallOp src, Operation* dst) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_9(mht_9_v, 362, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "CopyNonSymbolRefAttrs");

  NamedAttrList attrs;
  CollectAllowedAttrs(src, &attrs);

  for (auto& attr : attrs) {
    dst->setAttr(attr.getName(), attr.getValue());
  }

  return success();
}

void PropagateAttrsToOperations(CallOp src, Block::iterator begin,
                                Block::iterator end) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSutilsPSutilsDTcc mht_10(mht_10_v, 377, "", "./tensorflow/compiler/mlir/tfr/utils/utils.cc", "PropagateAttrsToOperations");

  // Find all the attributes in the call op. These attributes are not in the
  // op definition, so needs to be propagated to all the target ops.
  NamedAttrList attrs;
  CollectAllowedAttrs(src, &attrs);

  // Add all the attributes to the operations in the range.
  if (!attrs.empty()) {
    AddAttributes(begin, end, attrs);
  }
}

}  // namespace TFR
}  // namespace mlir
