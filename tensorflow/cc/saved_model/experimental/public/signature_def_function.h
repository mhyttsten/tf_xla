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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_
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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh() {
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


#include <vector>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/experimental/saved_model/public/signature_def_function.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/saved_model/experimental/public/signature_def_function_metadata.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// SignatureDefFunctions are functions that correspond to either:
// "signatures" saved from a TF2 SavedModel APIs:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/python/saved_model/save.py#L830-L854
// Or the "SignatureDefMap" saved from TF1 SavedModel APIs:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/python/saved_model/load_v1_in_v2_test.py#L170-L174
// In both cases, a SignatureDef is serialized as a SignatureDef protobuf:
// https://github.com/tensorflow/tensorflow/blob/8ce0600f58ed84a8c84a7bbdb014d1f09e44f4c8/tensorflow/core/protobuf/meta_graph.proto#L260-L330
// and represents a computation defined by a TF subgraph.
// These Signatures were primarily designed to be interoperable with the legacy
// TF 1 Session-based C++ SavedModelBundle loading APIs:
// https://github.com/tensorflow/tensorflow/blob/26c4ee0c833e74f94d0102d8b005c41a28b44445/tensorflow/cc/saved_model/loader.h#L96-L108
// SignatureDefFunctions have different semantics from regular TF2
// ConcreteFunctions, and are mainly intended provide a serving-friendly
// transition point from the TF1 Session API.
// First, SignatureDefFunctions have different calling conventions.
// SignatureDefFunctions' inputs and outputs are constrained to **flattened
// lists of TensorHandles only**. They do not support more exotic input/output
// types (like optionals, generators, etc). Additionally, this flattening means
// they will not preserve the exact interface of the original tf.function they
// were traced from, as things like composite tensors decay into their
// internal dense tensor representation.
// Second, all inputs and outputs are "named", and these names are load bearing
// (eg: they are part of the interface of tensorflow_serving):
// https://github.com/tensorflow/serving/blob/e0d247b2e4050713194b8fad0be24a0636df7209/tensorflow_serving/apis/predict.proto#L21
// https://github.com/tensorflow/serving/blob/e0d247b2e4050713194b8fad0be24a0636df7209/tensorflow_serving/apis/predict.proto#L39
// The name of each input/output is stored in the corresponding tf::Argument in
// SignatureDefFunctionMetadata::arguments(). Users must ensure the order of
// TensorHandles passed to the function matches with the order of named
// arguments. Similarly the name of the outputs is stored in
// SignatureDefFunctionMetadata::returns().
class SignatureDefFunction final {
 public:
  // Returns FunctionMetadata associated with this ConcreteFunction.
  const SignatureDefFunctionMetadata* GetFunctionMetadata();

 private:
  friend class SavedModelAPI;
  friend class ConcreteFunctionList;

  // TODO(bmzhao): Consider adding a macro for wrapping/unwrapping
  // when moving out of experimental.
  static SignatureDefFunction* wrap(TF_SignatureDefFunction* p) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh mht_0(mht_0_v, 240, "", "./tensorflow/cc/saved_model/experimental/public/signature_def_function.h", "wrap");

    return reinterpret_cast<SignatureDefFunction*>(p);
  }
  static TF_SignatureDefFunction* unwrap(SignatureDefFunction* p) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh mht_1(mht_1_v, 246, "", "./tensorflow/cc/saved_model/experimental/public/signature_def_function.h", "unwrap");

    return reinterpret_cast<TF_SignatureDefFunction*>(p);
  }
};

inline const SignatureDefFunctionMetadata*
SignatureDefFunction::GetFunctionMetadata() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsignature_def_functionDTh mht_2(mht_2_v, 255, "", "./tensorflow/cc/saved_model/experimental/public/signature_def_function.h", "SignatureDefFunction::GetFunctionMetadata");

  return SignatureDefFunctionMetadata::wrap(
      TF_SignatureDefFunctionGetMetadata(unwrap(this)));
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SIGNATURE_DEF_FUNCTION_H_
