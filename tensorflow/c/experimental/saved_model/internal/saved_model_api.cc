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
class MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc() {
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

#include "tensorflow/c/experimental/saved_model/public/saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "absl/types/optional.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/experimental/saved_model/core/saved_model_api.h"
#include "tensorflow/c/experimental/saved_model/core/tf_saved_model_api.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_list_type.h"
#include "tensorflow/c/experimental/saved_model/internal/concrete_function_type.h"
#include "tensorflow/c/experimental/saved_model/internal/saved_model_api_type.h"
#include "tensorflow/c/experimental/saved_model/internal/signature_def_function_type.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

extern "C" {

TF_SavedModel* TF_LoadSavedModel(const char* dirname, TFE_Context* ctx,
                                 TF_Status* status) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc mht_0(mht_0_v, 210, "", "./tensorflow/c/experimental/saved_model/internal/saved_model_api.cc", "TF_LoadSavedModel");

  std::string saved_model_dir(dirname);
  std::unique_ptr<tensorflow::SavedModelAPI> result;

  if (tensorflow::unwrap(ctx)->UsesTFRT()) {
    status->status = tensorflow::errors::Unimplemented(
        "TFRT SavedModel implementation will be added in the future");
  } else {
    std::unique_ptr<tensorflow::TFSavedModelAPI> saved_model;
    status->status = tensorflow::TFSavedModelAPI::Load(
        dirname, absl::nullopt,
        tensorflow::down_cast<tensorflow::EagerContext*>(
            tensorflow::unwrap(ctx)),
        &saved_model);
    result = std::move(saved_model);
  }

  if (!status->status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result.release());
}

TF_SavedModel* TF_LoadSavedModelWithTags(const char* dirname, TFE_Context* ctx,
                                         const char* const* tags, int tags_len,
                                         TF_Status* status) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc mht_1(mht_1_v, 239, "", "./tensorflow/c/experimental/saved_model/internal/saved_model_api.cc", "TF_LoadSavedModelWithTags");

  std::string saved_model_dir(dirname);

  std::unordered_set<std::string> tagset;
  for (int i = 0; i < tags_len; ++i) {
    tagset.insert(std::string(tags[i]));
  }

  std::unique_ptr<tensorflow::SavedModelAPI> result;
  if (tensorflow::unwrap(ctx)->UsesTFRT()) {
    status->status = tensorflow::errors::Unimplemented(
        "TFRT SavedModel implementation will be added in the future");
  } else {
    std::unique_ptr<tensorflow::TFSavedModelAPI> saved_model;
    status->status = tensorflow::TFSavedModelAPI::Load(
        dirname, tagset,
        tensorflow::down_cast<tensorflow::EagerContext*>(
            tensorflow::unwrap(ctx)),
        &saved_model);
    result = std::move(saved_model);
  }

  if (!status->status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result.release());
}

void TF_DeleteSavedModel(TF_SavedModel* model) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc mht_2(mht_2_v, 270, "", "./tensorflow/c/experimental/saved_model/internal/saved_model_api.cc", "TF_DeleteSavedModel");

  delete tensorflow::unwrap(model);
}

TF_ConcreteFunction* TF_GetSavedModelConcreteFunction(TF_SavedModel* model,
                                                      const char* function_path,
                                                      TF_Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("function_path: \"" + (function_path == nullptr ? std::string("nullptr") : std::string((char*)function_path)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc mht_3(mht_3_v, 280, "", "./tensorflow/c/experimental/saved_model/internal/saved_model_api.cc", "TF_GetSavedModelConcreteFunction");

  tensorflow::ConcreteFunction* result = nullptr;
  tensorflow::Status get_function_status =
      tensorflow::unwrap(model)->GetFunction(function_path, &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result);
}

TF_CAPI_EXPORT extern TF_SignatureDefFunction*
TF_GetSavedModelSignatureDefFunction(TF_SavedModel* model,
                                     const char* signature_def_key,
                                     TF_Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("signature_def_key: \"" + (signature_def_key == nullptr ? std::string("nullptr") : std::string((char*)signature_def_key)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSsaved_modelPSinternalPSsaved_model_apiDTcc mht_4(mht_4_v, 298, "", "./tensorflow/c/experimental/saved_model/internal/saved_model_api.cc", "TF_GetSavedModelSignatureDefFunction");

  tensorflow::SignatureDefFunction* result = nullptr;
  tensorflow::Status get_function_status =
      tensorflow::unwrap(model)->GetSignatureDefFunction(signature_def_key,
                                                         &result);
  status->status.Update(get_function_status);
  if (!get_function_status.ok()) {
    return nullptr;
  }
  return tensorflow::wrap(result);
}

}  // end extern "C"
