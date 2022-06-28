/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_
#define TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTh() {
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


#include <functional>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace checkpoint {

ABSL_CONST_INIT extern const absl::string_view
    kCheckpointCallbackManagerResourceName;

// StatusOr<std::string> save_callback(absl::string_view checkpoint_id);
using SaveCallback = std::function<StatusOr<std::string>(absl::string_view)>;

// Status restore_callback(absl::string_view checkpoint_id,
//                         absl::string_view content_from_checkpoint);
using RestoreCallback =
    std::function<Status(absl::string_view, absl::string_view)>;

// A class to save and restore additional information for checkpointing.
class CheckpointCallbackManager : public ResourceBase {
 public:
  CheckpointCallbackManager() = default;

  // Not copyable or movable
  CheckpointCallbackManager(const CheckpointCallbackManager&) = delete;
  CheckpointCallbackManager& operator=(const CheckpointCallbackManager&) =
      delete;

  std::string DebugString() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTh mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/checkpoint_callback_manager.h", "DebugString");

    return "CheckpointCallbackManager";
  }

  // Infers a checkpoint id and directory from a prefix
  // passed to SaveV2 / RestoreV2 Ops
  static StatusOr<std::pair<std::string, std::string>>
  GetCheckpointIdAndPathFromPrefix(absl::string_view prefix);

  // Register a save callback.
  // The passed callback will be triggered with an identified checkpoint id.
  // The callback should return a string content needs to be stored
  // as a part of a checkpoint, and then the content is stored as a file
  // with the registered the file_extension.
  Status RegisterSaveCallback(absl::string_view file_extension,
                              SaveCallback callback);

  // Checks if a registered save callback exists for an extension.
  bool DoesSaveCallbackExist(absl::string_view file_extension);

  // Register a restore callback.
  // The passed file_extension is used to generate a file name together with
  // an identified checkpoint_id. If the file exists, the registered callback
  // is triggered with the content of the file.
  Status RegisterRestoreCallback(absl::string_view file_extension,
                                 RestoreCallback callback);

  // Checks if a registered restore callback exists for an extension.
  bool DoesRestoreCallbackExist(absl::string_view file_extension);

  // Should be triggered from SaveV2()::Compute().
  void Save(absl::string_view prefix);

  // Should be triggered from RestoreV2()::Compute().
  void Restore(absl::string_view prefix);

 private:
  mutable mutex mu_;

  absl::flat_hash_map<std::string, SaveCallback> save_callbacks_
      TF_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, RestoreCallback> restore_callbacks_
      TF_GUARDED_BY(mu_);

  // Checkpoint save and restore could happen before save / restore callbacks
  // are registered. The last checkpoint information is kept in these variables
  // to trigger the registered callback lazily.
  std::pair<std::string, std::string> last_restored_checkpoint_id_and_dir_
      TF_GUARDED_BY(mu_);

  std::pair<std::string, std::string> last_saved_checkpoint_id_and_dir_
      TF_GUARDED_BY(mu_);
};

}  // namespace checkpoint
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CHECKPOINT_CALLBACK_MANAGER_H_
