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
class MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc() {
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
#include "tensorflow/core/kernels/checkpoint_callback_manager.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace checkpoint {

const absl::string_view kCheckpointCallbackManagerResourceName =
    "checkpoint_callback_manager";

namespace {

const absl::string_view kCheckpointFileRegex = "^part-[0-9]*-of-[0-9]*$";
const absl::string_view kCheckpointTempDirRegex = "-[0-9]*_temp$";
const absl::string_view kCheckpointDirRegex = "-[0-9]*$";
const absl::string_view kCheckpointTempDirSuffix = "_temp";

void TriggerSaveCallbackIfFileNotExist(absl::string_view checkpoint_id,
                                       absl::string_view checkpoint_dir,
                                       absl::string_view file_extension,
                                       SaveCallback callback) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("checkpoint_id: \"" + std::string(checkpoint_id.data(), checkpoint_id.size()) + "\"");
   mht_0_v.push_back("checkpoint_dir: \"" + std::string(checkpoint_dir.data(), checkpoint_dir.size()) + "\"");
   mht_0_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "TriggerSaveCallbackIfFileNotExist");

  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));

  // If the file already exists, we are done.
  if (Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  LOG(INFO) << "Calling a save callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  // The callback should return a string to store.
  StatusOr<std::string> save_content = callback(checkpoint_id);
  if (!save_content.ok()) {
    LOG(WARNING) << save_content.status();
    return;
  }

  Status write_status =
      WriteStringToFile(Env::Default(), file_path, *save_content);
  if (!write_status.ok()) {
    LOG(WARNING) << write_status;
  } else {
    LOG(INFO) << "A CheckpointCallbackManager has been written to "
              << file_path;
  }
}

void TriggerRestoreCallbackIfFileExists(absl::string_view checkpoint_id,
                                        absl::string_view checkpoint_dir,
                                        absl::string_view file_extension,
                                        RestoreCallback callback) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("checkpoint_id: \"" + std::string(checkpoint_id.data(), checkpoint_id.size()) + "\"");
   mht_1_v.push_back("checkpoint_dir: \"" + std::string(checkpoint_dir.data(), checkpoint_dir.size()) + "\"");
   mht_1_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_1(mht_1_v, 257, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "TriggerRestoreCallbackIfFileExists");

  const std::string file_path = io::JoinPath(
      checkpoint_dir, absl::StrCat(checkpoint_id, ".", file_extension));
  if (!Env::Default()->FileExists(file_path).ok()) {
    return;
  }
  std::string payload;
  Status read_status = ReadFileToString(Env::Default(), file_path, &payload);
  if (!read_status.ok()) {
    LOG(WARNING) << "Failed to read: " << read_status;
    return;
  }

  LOG(INFO) << "Calling a restore callback: file_extension = " << file_extension
            << ", checkpoint_id = " << checkpoint_id;
  Status callback_status = callback(checkpoint_id, payload);
  if (!callback_status.ok()) {
    LOG(WARNING) << callback_status;
  }
}

}  // namespace

//  Examples:
//    "/foo/bar/checkpoint-1_temp/part-00000-of-00001" -->
//        ("checkpoint-1", "/foo/bar");
//    "/foo/bar/checkpoint-2/part-00000-of-00001" -->
//        ("checkpoint-2", "/foo/bar");
//    "/foo/bar/checkpoint-3" --> ("checkpoint-3", "/foo/bar");
//    "/foo/bar"              --> NotFound error
StatusOr<std::pair<std::string, std::string>>
CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix(
    absl::string_view prefix) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_2(mht_2_v, 293, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::GetCheckpointIdAndPathFromPrefix");

  for (absl::string_view path = prefix;; path = io::Dirname(path)) {
    absl::string_view basename = io::Basename(path);

    // Failed to find checkpoint_id
    if (basename.empty()) break;

    // Skip known checkpoint file: e.g., part-00000-of-00001
    if (RE2::PartialMatch(basename, kCheckpointFileRegex)) continue;

    // With _temp suffix: e.g., checkpoint-1_temp
    if (RE2::PartialMatch(basename, kCheckpointTempDirRegex)) {
      // Trim suffix, "_temp".
      return std::make_pair(
          std::string(basename.substr(
              0, basename.length() - kCheckpointTempDirSuffix.length())),
          std::string(io::Dirname(path)));
    }

    // Without _temp suffix: e.g., checkpoint-1
    if (RE2::PartialMatch(basename, kCheckpointDirRegex)) {
      return std::make_pair(std::string(basename),
                            std::string(io::Dirname(path)));
    }
  }
  return errors::NotFound(
      absl::StrCat("Failed to find a checkpoint id. prefix = ", prefix));
}

Status CheckpointCallbackManager::RegisterSaveCallback(
    absl::string_view file_extension, SaveCallback callback) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_3(mht_3_v, 327, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::RegisterSaveCallback");

  SaveCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!save_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }

    // If last_saved_checkpoint_id_and_dir_ is not empty,
    // tries to trigger save callback lazily.
    if (!last_saved_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = save_callbacks_[file_extension];
      checkpoint_id = last_saved_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_saved_checkpoint_id_and_dir_.second;
    }
  }

  if (lazy_callback != nullptr) {
    TriggerSaveCallbackIfFileNotExist(checkpoint_id, checkpoint_dir,
                                      file_extension, lazy_callback);
  }
  return Status::OK();
}

bool CheckpointCallbackManager::DoesSaveCallbackExist(
    absl::string_view file_extension) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_4(mht_4_v, 359, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::DoesSaveCallbackExist");

  tf_shared_lock l(mu_);
  return save_callbacks_.contains(file_extension);
}

Status CheckpointCallbackManager::RegisterRestoreCallback(
    absl::string_view file_extension, RestoreCallback callback) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_5(mht_5_v, 369, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::RegisterRestoreCallback");

  RestoreCallback lazy_callback = nullptr;
  std::string checkpoint_id;
  std::string checkpoint_dir;
  {
    mutex_lock l(mu_);
    if (!restore_callbacks_.try_emplace(file_extension, std::move(callback))
             .second) {
      return errors::AlreadyExists("A callback already exists.");
    }

    // If last_restored_checkpoint_id_and_dir_ is not empty,
    // tries to trigger restore callback lazily.
    if (!last_restored_checkpoint_id_and_dir_.first.empty()) {
      lazy_callback = restore_callbacks_[file_extension];
      checkpoint_id = last_restored_checkpoint_id_and_dir_.first;
      checkpoint_dir = last_restored_checkpoint_id_and_dir_.second;
    }
  }

  if (lazy_callback != nullptr) {
    TriggerRestoreCallbackIfFileExists(checkpoint_id, checkpoint_dir,
                                       file_extension, lazy_callback);
  }
  return Status::OK();
}

bool CheckpointCallbackManager::DoesRestoreCallbackExist(
    absl::string_view file_extension) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file_extension: \"" + std::string(file_extension.data(), file_extension.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_6(mht_6_v, 401, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::DoesRestoreCallbackExist");

  tf_shared_lock l(mu_);
  return restore_callbacks_.contains(file_extension);
}

void CheckpointCallbackManager::Save(absl::string_view prefix) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_7(mht_7_v, 410, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::Save");

  StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }

  // Create a copy to avoid holding lock while calling a callback.
  absl::flat_hash_map<std::string, SaveCallback> copy_of_save_callbacks;
  {
    mutex_lock l(mu_);
    last_saved_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_save_callbacks = save_callbacks_;
  }

  for (const auto& name_and_callback : copy_of_save_callbacks) {
    TriggerSaveCallbackIfFileNotExist(id_and_dir->first, id_and_dir->second,
                                      name_and_callback.first,
                                      name_and_callback.second);
  }
}

void CheckpointCallbackManager::Restore(absl::string_view prefix) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScheckpoint_callback_managerDTcc mht_8(mht_8_v, 436, "", "./tensorflow/core/kernels/checkpoint_callback_manager.cc", "CheckpointCallbackManager::Restore");

  StatusOr<std::pair<std::string, std::string>> id_and_dir =
      GetCheckpointIdAndPathFromPrefix(prefix);
  if (!id_and_dir.ok()) {
    return;
  }

  // Create a copy to avoid holding lock while calling a callback.
  absl::flat_hash_map<std::string, RestoreCallback> copy_of_restore_callbacks;
  {
    mutex_lock l(mu_);
    last_restored_checkpoint_id_and_dir_ = *id_and_dir;
    copy_of_restore_callbacks = restore_callbacks_;
  }

  for (const auto& name_and_callback : copy_of_restore_callbacks) {
    TriggerRestoreCallbackIfFileExists(id_and_dir->first, id_and_dir->second,
                                       name_and_callback.first,
                                       name_and_callback.second);
  }
}

}  // namespace checkpoint
}  // namespace tensorflow
