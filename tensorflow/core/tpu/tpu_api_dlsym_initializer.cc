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
class MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc() {
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

#include "tensorflow/core/tpu/tpu_api_dlsym_initializer.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/tpu_api_dlsym_set_fn.h"

#if !defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/cloud/gcs_file_system.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_initializer_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#endif


// Reminder: Update tpu_library_loader_windows.cc if you are adding new publicly
// visible methods.

namespace tensorflow {
namespace tpu {
namespace {
#if defined(PLATFORM_GOOGLE)
Status InitializeTpuLibrary(void* library_handle) {
  return errors::Unimplemented("You must statically link in a TPU library.");
}
#else  // PLATFORM_GOOGLE
#include "tensorflow/core/tpu/tpu_library_init_fns.inc"

Status InitializeTpuLibrary(void* library_handle) {
  Status s = InitializeTpuStructFns(library_handle);

  // Retrieve arguments from environment if applicable
  std::pair<std::vector<std::string>, std::vector<const char*> > args =
      GetLibTpuInitArguments();

  // TPU platform registration must only be performed after the library is
  // loaded. We do not want to register a TPU platform in XLA without the
  // supporting library providing the necessary APIs.
  if (s.ok()) {
    void (*initialize_fn)(bool init_library, int num_args, const char** args);
    initialize_fn = reinterpret_cast<decltype(initialize_fn)>(
        dlsym(library_handle, "TfTpu_Initialize"));
    (*initialize_fn)(/*init_library=*/true, args.second.size(),
                     args.second.data());

    RegisterTpuPlatform();
  }

  return s;
}

void* CreateGcsFilesystemFn() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc mht_0(mht_0_v, 243, "", "./tensorflow/core/tpu/tpu_api_dlsym_initializer.cc", "CreateGcsFilesystemFn");

  return new tensorflow::RetryingGcsFileSystem();
}

// This is a temporary fix for including GCS file system on TPU builds.
// Will be removed once b/176954917 is fully resolved with the build fix.
void InitializeCreateGcsFileSystemFnPtr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc mht_1(mht_1_v, 252, "", "./tensorflow/core/tpu/tpu_api_dlsym_initializer.cc", "InitializeCreateGcsFileSystemFnPtr");

  int fd = shm_open(absl::StrCat("/tmp_tf_gcs_fs_pointer_", getpid()).data(),
                    O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(ERROR) << "Unable to open shared memory for GCS file system creator.";
    return;
  }

  if (ftruncate(fd, sizeof(tensorflow::FileSystem*)) == -1) {
    LOG(ERROR)
        << "Unable to allocate shared memory for GCS file system creator.";
    return;
  }

  void* (**fn)() = reinterpret_cast<void* (**)()>(mmap(
      NULL, sizeof(void* (*)()), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (fn == MAP_FAILED) {
    LOG(ERROR) << "Cannot mmap shared memory for GCS file system creator.";
    return;
  }

  *fn = &CreateGcsFilesystemFn;

  munmap(fn, sizeof(void* (*)()));
  close(fd);

  // Clean up shared memory on a clean exit.
  atexit([]() {
    shm_unlink(absl::StrCat("/tmp_tf_gcs_fs_pointer_", getpid()).data());
  });
}

bool FindAndLoadTpuLibrary() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_api_dlsym_initializerDTcc mht_2(mht_2_v, 287, "", "./tensorflow/core/tpu/tpu_api_dlsym_initializer.cc", "FindAndLoadTpuLibrary");

  const char* env_value = getenv("TPU_LIBRARY_PATH");
  const char* libtpu_path =
      env_value && strlen(env_value) > 0 ? env_value : "libtpu.so";
  LOG(INFO) << "Libtpu path is: " << libtpu_path;
  void* library = dlopen(libtpu_path, RTLD_NOW);
  if (library) {
    // We can open the shared library which means we are in a TPU environment.
    // Try to acquire exclusive access.
    if (TryAcquireTpuLock()) {
      InitializeTpuLibrary(library);
    }
  }

  InitializeCreateGcsFileSystemFnPtr();
  return true;
}

static bool tpu_library_finder = FindAndLoadTpuLibrary();
#endif  // PLATFORM_GOOGLE
}  // namespace
}  // namespace tpu
}  // namespace tensorflow
