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
class MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc() {
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

#include "tensorflow/core/tpu/tpu_initializer_helper.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tpu {
namespace {

static std::string GetEnvVar(const char* name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/tpu/tpu_initializer_helper.cc", "GetEnvVar");

  // Constructing a std::string directly from nullptr is undefined behavior.
  return absl::StrCat(getenv(name));
}

bool GetEnvBool(const char* name, bool defval) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/tpu/tpu_initializer_helper.cc", "GetEnvBool");

  const char* env = getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  int int_env;
  bool has_int = absl::SimpleAtoi(env, &int_env);
  return has_int && int_env != 0;
}

}  // namespace

// This function gets pid of a process and checks if that process is using tpu.
// It is not able to check processes that are owned by another user.
bool IsTpuUsed(int64_t pid) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/tpu/tpu_initializer_helper.cc", "IsTpuUsed");

  std::string path = absl::StrCat("/proc/", pid, "/fd");
  DIR* raw_fd_dir = opendir(path.c_str());
  if (!raw_fd_dir) {
    return false;
  }
  std::unique_ptr<DIR, int (*)(DIR*)> fd_dir(raw_fd_dir, closedir);
  struct dirent* ent;
  std::string line;
  std::string tpu_dev_path = "/dev/accel0";
  line.resize(tpu_dev_path.size());
  while ((ent = readdir(raw_fd_dir))) {
    if (!isdigit(*ent->d_name)) continue;
    int64_t fd = strtol(ent->d_name, nullptr, 10);
    path = absl::StrCat("/proc/", pid, "/fd/", fd);
    if (!readlink(path.c_str(), &line[0], line.size())) continue;
    if (line != tpu_dev_path) continue;
    return true;
  }
  return false;
}

// This function iterates through all the processes in /proc and logs if any
// process it was able to check is using the TPU. It does not have permission to
// processes owned by another user.
// TODO (shahrokhi) use tensorflow/core/platform/filesystem (GetChildren) for
// this.
bool FindAndLogLibtpuProcess() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/tpu/tpu_initializer_helper.cc", "FindAndLogLibtpuProcess");

  DIR* proc = opendir("/proc");

  if (proc == nullptr) {
    return false;
  }
  std::unique_ptr<DIR, int (*)(DIR*)> proc_dir(proc, closedir);
  struct dirent* ent;
  int64_t pid;
  while ((ent = readdir(proc))) {
    if (!isdigit(*ent->d_name)) continue;

    pid = strtol(ent->d_name, nullptr, 10);
    if (IsTpuUsed(pid)) {
      LOG(INFO) << "libtpu.so is already in use by process with pid " << pid
                << ". Not attempting to load libtpu.so in this process.";
      return true;
    }
  }
  return false;
}

bool TryAcquireTpuLock() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_initializer_helperDTcc mht_4(mht_4_v, 291, "", "./tensorflow/core/tpu/tpu_initializer_helper.cc", "TryAcquireTpuLock");

  static absl::Mutex* mu = new absl::Mutex();
  absl::MutexLock l(mu);

  static bool attempted_file_open = false;
  static bool should_load_library = false;

  if (!attempted_file_open) {
    std::string load_library_override =
        absl::StrCat(getenv("TPU_LOAD_LIBRARY"));

    if (load_library_override == "1") {
      return true;
    } else if (load_library_override == "0") {
      return false;
    }
    should_load_library = true;

    // If TPU_CHIPS_PER_PROCESS_BOUNDS doesn't include all chips, we assume
    // we're using different chips in different processes and thus multiple
    // libtpu loads are ok.
    // TODO(skyewm): we could make per-chip lock files and look at
    // TPU_VISIBLE_DEVICES if we wanted to make this really precise.
    std::string chips_per_process_bounds =
        GetEnvVar("TPU_CHIPS_PER_PROCESS_BOUNDS");
    bool allow_multiple_libtpu_load =
        GetEnvBool("ALLOW_MULTIPLE_LIBTPU_LOAD", false);
    // TODO(skyewm): remove this when TPU_CHIPS_PER_HOST_BOUNDS is fully
    // deprecated
    if (chips_per_process_bounds.empty()) {
      chips_per_process_bounds = GetEnvVar("TPU_CHIPS_PER_HOST_BOUNDS");
    }
    if ((chips_per_process_bounds.empty() ||
         chips_per_process_bounds == "2,2,1") &&
        !allow_multiple_libtpu_load) {
      int fd = open("/tmp/libtpu_lockfile", O_CREAT | O_RDWR, 0644);

      // This lock is held until the process exits intentionally. The underlying
      // TPU device will be held on until it quits.
      if (lockf(fd, F_TLOCK, 0) != 0) {
        if (!FindAndLogLibtpuProcess()) {
          LOG(INFO) << "libtpu.so already in use by another process probably"
                       " owned by another user. "
                       "Run \"$ sudo lsof -w /dev/accel0\" to figure out "
                       "which process is using the TPU. Not "
                       "attempting to load libtpu.so in this process.";
        }
        should_load_library = false;
      } else {
        should_load_library = true;
      }
    } else {
      VLOG(1) << "TPU_CHIPS_PER_PROCESS_BOUNDS is not empty or "
                 "ALLOW_MULTIPLE_LIBTPU_LOAD is set to True, "
                 "therefore allowing multiple libtpu.so loads.";
      should_load_library = true;
    }
  }

  return should_load_library;
}

std::pair<std::vector<std::string>, std::vector<const char*>>
GetLibTpuInitArguments() {
  // We make copies of the arguments returned by getenv because the memory
  // returned may be altered or invalidated by further calls to getenv.
  std::vector<std::string> args;
  std::vector<const char*> arg_ptrs;

  // Retrieve arguments from environment if applicable.
  char* env = getenv("LIBTPU_INIT_ARGS");
  if (env != nullptr) {
    // TODO(frankchn): Handles quotes properly if necessary.
    args = absl::StrSplit(env, ' ');
  }

  arg_ptrs.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    arg_ptrs.push_back(args[i].data());
  }

  return {std::move(args), std::move(arg_ptrs)};
}

}  // namespace tpu
}  // namespace tensorflow
