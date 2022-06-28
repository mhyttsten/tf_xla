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

#ifndef TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_
#define TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_
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
class MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh {
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
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh() {
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


#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/c/experimental/saved_model/public/saved_model_api.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/cc/saved_model/experimental/public/concrete_function.h"
#include "tensorflow/cc/saved_model/experimental/public/concrete_function_list.h"
#include "tensorflow/cc/saved_model/experimental/public/signature_def_function.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// SavedModelAPI offers a way to load Tensorflow Saved Models
// (https://www.tensorflow.org/guide/saved_model) and execute saved
// tf.functions or legacy SignatureDefs in a TF2-idiomatic fashion.
// See RFC 207
// (https://github.com/tensorflow/community/blob/master/rfcs/20200218-tf-c-saved-model.md)
// TODO(bmzhao): Add an e2e example here, once ConcreteFunction::Run is added.
class SavedModelAPI {
 public:
  // Load a SavedModel from `dirname`.
  //
  // Params:
  //  saved_model_path - A directory filepath that the SavedModel is at.
  //  runtime - A runtime used to load SavedModelAPI. `runtime` must outlive the
  //            returned TF_SavedModel pointer.
  //  tags - Optional set of tags. If tags = nullptr, we expect the SavedModel
  //         to contain a single Metagraph (as for those exported from TF2's
  //         `tf.saved_model.save`). If tags != nullptr, we load the metagraph
  //         matching the tags:
  //         https://github.com/tensorflow/tensorflow/blob/428cdeda09aef81e958eeb274b83d27ad635b57b/tensorflow/core/protobuf/meta_graph.proto#L50-L56
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr.
  static std::unique_ptr<SavedModelAPI> Load(
      const std::string& saved_model_path, const Runtime& runtime,
      Status* status, const std::unordered_set<std::string>* tags = nullptr);

  // Retrieve a function from the TF2 SavedModel via function path.
  //
  // Params:
  //  function_path - A string containing the path from the root saved python
  //                  object to a tf.function method.
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  tensorflow::cc::ConcreteFunction pointer. The lifetime of this pointer
  //  is bound to SavedModelAPI it was loaded from.
  ConcreteFunction* GetConcreteFunction(const std::string& function_path,
                                        Status* status);

  // Retrieve a function from the TF SavedModel via a SignatureDef key.
  //
  // Params:
  //  signature_def_key - String key of SignatureDef map of a SavedModel:
  //                      https://github.com/tensorflow/tensorflow/blob/69b08900b1e991d84bce31f3b404f5ed768f339f/tensorflow/core/protobuf/meta_graph.proto#L89
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  tensorflow::cc::ConcreteFunction pointer. The lifetime of this pointer
  //  is bound to SavedModelAPI it was loaded from.
  SignatureDefFunction* GetSignatureDefFunction(
      const std::string& function_path, Status* status);

  // SavedModelAPI is movable, but not copyable.
  SavedModelAPI(SavedModelAPI&&) = default;
  SavedModelAPI& operator=(SavedModelAPI&&) = default;

 private:
  SavedModelAPI(const SavedModelAPI&) = delete;
  SavedModelAPI& operator=(const SavedModelAPI&) = delete;

  explicit SavedModelAPI(TF_SavedModel* model) : saved_model_(model) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh mht_0(mht_0_v, 264, "", "./tensorflow/cc/saved_model/experimental/public/saved_model_api.h", "SavedModelAPI");
}
  struct TFSavedModelDeleter {
    void operator()(TF_SavedModel* p) const { TF_DeleteSavedModel(p); }
  };
  std::unique_ptr<TF_SavedModel, TFSavedModelDeleter> saved_model_;
};

inline std::unique_ptr<SavedModelAPI> SavedModelAPI::Load(
    const std::string& saved_model_path, const Runtime& runtime, Status* status,
    const std::unordered_set<std::string>* tags) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("saved_model_path: \"" + saved_model_path + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh mht_1(mht_1_v, 277, "", "./tensorflow/cc/saved_model/experimental/public/saved_model_api.h", "SavedModelAPI::Load");

  TF_SavedModel* saved_model = nullptr;

  if (tags == nullptr) {
    saved_model =
        TF_LoadSavedModel(saved_model_path.c_str(), runtime.GetTFEContext(),
                          status->GetTFStatus());
  } else {
    std::vector<const char*> tags_vector;
    tags_vector.reserve(tags->size());
    for (const std::string& tag : *tags) {
      tags_vector.push_back(tag.c_str());
    }
    saved_model = TF_LoadSavedModelWithTags(
        saved_model_path.c_str(), runtime.GetTFEContext(), tags_vector.data(),
        tags_vector.size(), status->GetTFStatus());
  }

  if (!status->ok()) {
    return nullptr;
  }

  // We can't use std::make_unique here because of its interaction with a
  // private constructor: https://abseil.io/tips/134
  return std::unique_ptr<SavedModelAPI>(new SavedModelAPI(saved_model));
}

inline ConcreteFunction* SavedModelAPI::GetConcreteFunction(
    const std::string& function_path, Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("function_path: \"" + function_path + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh mht_2(mht_2_v, 309, "", "./tensorflow/cc/saved_model/experimental/public/saved_model_api.h", "SavedModelAPI::GetConcreteFunction");

  TF_ConcreteFunction* function = TF_GetSavedModelConcreteFunction(
      saved_model_.get(), function_path.c_str(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  return ConcreteFunction::wrap(function);
}

inline SignatureDefFunction* SavedModelAPI::GetSignatureDefFunction(
    const std::string& function_path, Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("function_path: \"" + function_path + "\"");
   MHTracer_DTPStensorflowPSccPSsaved_modelPSexperimentalPSpublicPSsaved_model_apiDTh mht_3(mht_3_v, 323, "", "./tensorflow/cc/saved_model/experimental/public/saved_model_api.h", "SavedModelAPI::GetSignatureDefFunction");

  TF_SignatureDefFunction* function = TF_GetSavedModelSignatureDefFunction(
      saved_model_.get(), function_path.c_str(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  return SignatureDefFunction::wrap(function);
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_EXPERIMENTAL_PUBLIC_SAVED_MODEL_API_H_
