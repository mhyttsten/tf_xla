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
class MHTracer_DTPStensorflowPScPSenvDTcc {
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
   MHTracer_DTPStensorflowPScPSenvDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSenvDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/env.h"

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"

struct TF_StringStream {
  std::vector<::tensorflow::string>* list;
  size_t position;
};

void TF_CreateDir(const char* dirname, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_0(mht_0_v, 200, "", "./tensorflow/c/env.cc", "TF_CreateDir");

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->CreateDir(dirname));
}

void TF_DeleteDir(const char* dirname, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_1(mht_1_v, 210, "", "./tensorflow/c/env.cc", "TF_DeleteDir");

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->DeleteDir(dirname));
}

void TF_DeleteRecursively(const char* dirname, int64_t* undeleted_file_count,
                          int64_t* undeleted_dir_count, TF_Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_2(mht_2_v, 221, "", "./tensorflow/c/env.cc", "TF_DeleteRecursively");

  ::int64_t f, d;

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->DeleteRecursively(dirname, &f, &d));
  *undeleted_file_count = f;
  *undeleted_dir_count = d;
}

void TF_FileStat(const char* filename, TF_FileStatistics* stats,
                 TF_Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_3(mht_3_v, 236, "", "./tensorflow/c/env.cc", "TF_FileStat");

  ::tensorflow::FileStatistics cc_stats;
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Status s =
      ::tensorflow::Env::Default()->Stat(filename, &cc_stats);
  ::tensorflow::Set_TF_Status_from_Status(status, s);
  if (s.ok()) {
    stats->length = cc_stats.length;
    stats->mtime_nsec = cc_stats.mtime_nsec;
    stats->is_directory = cc_stats.is_directory;
  }
}

void TF_NewWritableFile(const char* filename, TF_WritableFileHandle** handle,
                        TF_Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_4(mht_4_v, 254, "", "./tensorflow/c/env.cc", "TF_NewWritableFile");

  std::unique_ptr<::tensorflow::WritableFile> f;
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Status s =
      ::tensorflow::Env::Default()->NewWritableFile(filename, &f);
  ::tensorflow::Set_TF_Status_from_Status(status, s);

  if (s.ok()) {
    *handle = reinterpret_cast<TF_WritableFileHandle*>(f.release());
  }
}

void TF_CloseWritableFile(TF_WritableFileHandle* handle, TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_5(mht_5_v, 269, "", "./tensorflow/c/env.cc", "TF_CloseWritableFile");

  auto* cc_file = reinterpret_cast<::tensorflow::WritableFile*>(handle);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, cc_file->Close());
  delete cc_file;
}

void TF_SyncWritableFile(TF_WritableFileHandle* handle, TF_Status* status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_6(mht_6_v, 279, "", "./tensorflow/c/env.cc", "TF_SyncWritableFile");

  auto* cc_file = reinterpret_cast<::tensorflow::WritableFile*>(handle);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, cc_file->Sync());
}

void TF_FlushWritableFile(TF_WritableFileHandle* handle, TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_7(mht_7_v, 288, "", "./tensorflow/c/env.cc", "TF_FlushWritableFile");

  auto* cc_file = reinterpret_cast<::tensorflow::WritableFile*>(handle);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(status, cc_file->Flush());
}

void TF_AppendWritableFile(TF_WritableFileHandle* handle, const char* data,
                           size_t length, TF_Status* status) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_8(mht_8_v, 299, "", "./tensorflow/c/env.cc", "TF_AppendWritableFile");

  auto* cc_file = reinterpret_cast<::tensorflow::WritableFile*>(handle);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, cc_file->Append(::tensorflow::StringPiece{data, length}));
}

void TF_DeleteFile(const char* filename, TF_Status* status) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("filename: \"" + (filename == nullptr ? std::string("nullptr") : std::string((char*)filename)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_9(mht_9_v, 310, "", "./tensorflow/c/env.cc", "TF_DeleteFile");

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->DeleteFile(filename));
}

bool TF_StringStreamNext(TF_StringStream* list, const char** result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_10(mht_10_v, 319, "", "./tensorflow/c/env.cc", "TF_StringStreamNext");

  if (list->position >= list->list->size()) {
    *result = nullptr;
    return false;
  }

  *result = list->list->at(list->position++).c_str();
  return true;
}

void TF_StringStreamDone(TF_StringStream* list) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_11(mht_11_v, 332, "", "./tensorflow/c/env.cc", "TF_StringStreamDone");

  delete list->list;
  delete list;
}
TF_StringStream* TF_GetChildren(const char* dirname, TF_Status* status) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("dirname: \"" + (dirname == nullptr ? std::string("nullptr") : std::string((char*)dirname)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_12(mht_12_v, 340, "", "./tensorflow/c/env.cc", "TF_GetChildren");

  auto* children = new std::vector<::tensorflow::string>;

  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->GetChildren(dirname, children));

  auto* list = new TF_StringStream;
  list->list = children;
  list->position = 0;
  return list;
}

TF_StringStream* TF_GetLocalTempDirectories() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_13(mht_13_v, 356, "", "./tensorflow/c/env.cc", "TF_GetLocalTempDirectories");

  auto* tmpdirs = new std::vector<::tensorflow::string>;

  ::tensorflow::Env::Default()->GetLocalTempDirectories(tmpdirs);

  auto* list = new TF_StringStream;
  list->list = tmpdirs;
  list->position = 0;
  return list;
}

char* TF_GetTempFileName(const char* extension) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("extension: \"" + (extension == nullptr ? std::string("nullptr") : std::string((char*)extension)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_14(mht_14_v, 371, "", "./tensorflow/c/env.cc", "TF_GetTempFileName");

  return strdup(::tensorflow::io::GetTempFilename(extension).c_str());
}

TF_CAPI_EXPORT extern uint64_t TF_NowNanos(void) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_15(mht_15_v, 378, "", "./tensorflow/c/env.cc", "TF_NowNanos");

  return ::tensorflow::Env::Default()->NowNanos();
}

// Returns the number of microseconds since the Unix epoch.
TF_CAPI_EXPORT extern uint64_t TF_NowMicros(void) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_16(mht_16_v, 386, "", "./tensorflow/c/env.cc", "TF_NowMicros");

  return ::tensorflow::Env::Default()->NowMicros();
}

// Returns the number of seconds since the Unix epoch.
TF_CAPI_EXPORT extern uint64_t TF_NowSeconds(void) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_17(mht_17_v, 394, "", "./tensorflow/c/env.cc", "TF_NowSeconds");

  return ::tensorflow::Env::Default()->NowSeconds();
}

void TF_DefaultThreadOptions(TF_ThreadOptions* options) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_18(mht_18_v, 401, "", "./tensorflow/c/env.cc", "TF_DefaultThreadOptions");

  options->stack_size = 0;
  options->guard_size = 0;
  options->numa_node = -1;
}

TF_Thread* TF_StartThread(const TF_ThreadOptions* options,
                          const char* thread_name, void (*work_func)(void*),
                          void* param) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("thread_name: \"" + (thread_name == nullptr ? std::string("nullptr") : std::string((char*)thread_name)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_19(mht_19_v, 413, "", "./tensorflow/c/env.cc", "TF_StartThread");

  ::tensorflow::ThreadOptions cc_options;
  cc_options.stack_size = options->stack_size;
  cc_options.guard_size = options->guard_size;
  cc_options.numa_node = options->numa_node;
  return reinterpret_cast<TF_Thread*>(::tensorflow::Env::Default()->StartThread(
      cc_options, thread_name, [=]() { (*work_func)(param); }));
}

void TF_JoinThread(TF_Thread* thread) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSenvDTcc mht_20(mht_20_v, 425, "", "./tensorflow/c/env.cc", "TF_JoinThread");

  // ::tensorflow::Thread joins on destruction
  delete reinterpret_cast<::tensorflow::Thread*>(thread);
}

void* TF_LoadSharedLibrary(const char* library_filename, TF_Status* status) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("library_filename: \"" + (library_filename == nullptr ? std::string("nullptr") : std::string((char*)library_filename)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_21(mht_21_v, 434, "", "./tensorflow/c/env.cc", "TF_LoadSharedLibrary");

  void* handle = nullptr;
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->LoadDynamicLibrary(library_filename,
                                                               &handle));
  return handle;
}

void* TF_GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              TF_Status* status) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("symbol_name: \"" + (symbol_name == nullptr ? std::string("nullptr") : std::string((char*)symbol_name)) + "\"");
   MHTracer_DTPStensorflowPScPSenvDTcc mht_22(mht_22_v, 448, "", "./tensorflow/c/env.cc", "TF_GetSymbolFromLibrary");

  void* symbol = nullptr;
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::Set_TF_Status_from_Status(
      status, ::tensorflow::Env::Default()->GetSymbolFromLibrary(
                  handle, symbol_name, &symbol));
  return symbol;
}
