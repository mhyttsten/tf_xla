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
class MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc() {
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

/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/rpc/client/save_profile.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/profiler_service.pb.h"
#include "tensorflow/core/profiler/utils/file_system_utils.h"

// Windows.h #defines ERROR, but it is also used in
// tensorflow/core/util/event.proto
#undef ERROR
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace profiler {
namespace {


constexpr char kProtoTraceFileName[] = "trace";
constexpr char kTfStatsHelperSuffix[] = "tf_stats_helper_result";

Status DumpToolData(absl::string_view run_dir, absl::string_view host,
                    const ProfileToolData& tool, std::ostream* os) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("run_dir: \"" + std::string(run_dir.data(), run_dir.size()) + "\"");
   mht_0_v.push_back("host: \"" + std::string(host.data(), host.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "DumpToolData");

  // Don't save the intermediate results for combining the per host tool data.
  if (absl::EndsWith(tool.name(), kTfStatsHelperSuffix)) return Status::OK();
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool.name()));
  TF_RETURN_IF_ERROR(WriteStringToFile(Env::Default(), path, tool.data()));
  if (os) {
    *os << "Dumped tool data for " << tool.name() << " to " << path << '\n';
  }
  return Status::OK();
}

Status WriteGzippedDataToFile(const std::string& filepath,
                              const std::string& data) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filepath: \"" + filepath + "\"");
   mht_1_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_1(mht_1_v, 244, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "WriteGzippedDataToFile");

  std::unique_ptr<WritableFile> file;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(filepath, &file));
  io::ZlibCompressionOptions options = io::ZlibCompressionOptions::GZIP();
  io::ZlibOutputBuffer buffer(file.get(), options.input_buffer_size,
                              options.output_buffer_size, options);
  TF_RETURN_IF_ERROR(buffer.Init());
  TF_RETURN_IF_ERROR(buffer.Append(data));
  TF_RETURN_IF_ERROR(buffer.Close());
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

Status GetOrCreateRunDir(const std::string& repository_root,
                         const std::string& run, std::string* run_dir,
                         std::ostream* os) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("repository_root: \"" + repository_root + "\"");
   mht_2_v.push_back("run: \"" + run + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "GetOrCreateRunDir");

  // Creates a directory to <repository_root>/<run>/.
  *run_dir = ProfilerJoinPath(repository_root, run);
  *os << "Creating directory: " << *run_dir << '\n';
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(*run_dir));
  return Status::OK();
}
}  // namespace

std::string GetTensorBoardProfilePluginDir(const std::string& logdir) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("logdir: \"" + logdir + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "GetTensorBoardProfilePluginDir");

  constexpr char kPluginName[] = "plugins";
  constexpr char kProfileName[] = "profile";
  return ProfilerJoinPath(logdir, kPluginName, kProfileName);
}

Status MaybeCreateEmptyEventFile(const std::string& logdir) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("logdir: \"" + logdir + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "MaybeCreateEmptyEventFile");

  // Suffix for an empty event file.  it should be kept in sync with
  // _EVENT_FILE_SUFFIX in tensorflow/python/eager/profiler.py.
  constexpr char kProfileEmptySuffix[] = ".profile-empty";
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(logdir));

  std::vector<std::string> children;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(logdir, &children));
  for (const std::string& child : children) {
    if (absl::EndsWith(child, kProfileEmptySuffix)) {
      return Status::OK();
    }
  }
  EventsWriter event_writer(ProfilerJoinPath(logdir, "events"));
  return event_writer.InitWithSuffix(kProfileEmptySuffix);
}

Status SaveProfile(const std::string& repository_root, const std::string& run,
                   const std::string& host, const ProfileResponse& response,
                   std::ostream* os) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("repository_root: \"" + repository_root + "\"");
   mht_5_v.push_back("run: \"" + run + "\"");
   mht_5_v.push_back("host: \"" + host + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_5(mht_5_v, 312, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "SaveProfile");

  if (response.tool_data().empty()) return Status::OK();
  std::string run_dir;
  TF_RETURN_IF_ERROR(GetOrCreateRunDir(repository_root, run, &run_dir, os));
  // Windows file names do not support colons.
  std::string hostname = absl::StrReplaceAll(host, {{":", "_"}});
  for (const auto& tool_data : response.tool_data()) {
    TF_RETURN_IF_ERROR(DumpToolData(run_dir, hostname, tool_data, os));
  }
  return Status::OK();
}

Status SaveGzippedToolData(const std::string& repository_root,
                           const std::string& run, const std::string& host,
                           const std::string& tool_name,
                           const std::string& data) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("repository_root: \"" + repository_root + "\"");
   mht_6_v.push_back("run: \"" + run + "\"");
   mht_6_v.push_back("host: \"" + host + "\"");
   mht_6_v.push_back("tool_name: \"" + tool_name + "\"");
   mht_6_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_6(mht_6_v, 335, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "SaveGzippedToolData");

  std::string run_dir;
  std::stringstream ss;
  Status status = GetOrCreateRunDir(repository_root, run, &run_dir, &ss);
  LOG(INFO) << ss.str();
  TF_RETURN_IF_ERROR(status);
  std::string host_prefix = host.empty() ? "" : absl::StrCat(host, ".");
  std::string path =
      ProfilerJoinPath(run_dir, absl::StrCat(host_prefix, tool_name));
  TF_RETURN_IF_ERROR(WriteGzippedDataToFile(path, data));
  LOG(INFO) << "Dumped gzipped tool data for " << tool_name << " to " << path;
  return Status::OK();
}

std::string GetCurrentTimeStampAsString() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSrpcPSclientPSsave_profileDTcc mht_7(mht_7_v, 352, "", "./tensorflow/core/profiler/rpc/client/save_profile.cc", "GetCurrentTimeStampAsString");

  return absl::FormatTime("%E4Y_%m_%d_%H_%M_%S", absl::Now(),
                          absl::LocalTimeZone());
}

}  // namespace profiler
}  // namespace tensorflow
