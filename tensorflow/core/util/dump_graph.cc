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
class MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc() {
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

// Helper functions for dumping Graphs, GraphDefs, and FunctionDefs to files for
// debugging.

#include "tensorflow/core/util/dump_graph.h"

#include <memory>
#include <unordered_map>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {

namespace {
using strings::StrCat;

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniqueFilename(string name, const string& suffix = ".pbtxt") {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/util/dump_graph.cc", "MakeUniqueFilename");

  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?' ||
        ch == '\\') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  string filename = name;
  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, suffix);
  return filename;
}

struct GraphDumperConfig {
  mutex mu;

  // The dumper and suffix configured.
  struct Config {
    bool IsSet() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/util/dump_graph.cc", "IsSet");
 return dumper != nullptr; }
    std::function<Status(const Graph& graph,
                         const FunctionLibraryDefinition* flib_def,
                         WritableFile*)>
        dumper = nullptr;
    string suffix = ".pbtxt";
  } config TF_GUARDED_BY(mu);

  // Returns whether a custom dumper is set.
  bool IsSet() TF_LOCKS_EXCLUDED(mu) {
    mutex_lock lock(mu);
    return config.IsSet();
  }
};

GraphDumperConfig& GetGraphDumperConfig() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/util/dump_graph.cc", "GetGraphDumperConfig");

  static GraphDumperConfig config;
  return config;
}

// WritableFile that simply prints to stderr.
class StderrWritableFile : public WritableFile {
 public:
  StderrWritableFile() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_3(mht_3_v, 276, "", "./tensorflow/core/util/dump_graph.cc", "StderrWritableFile");
}

  Status Append(StringPiece data) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/util/dump_graph.cc", "Append");

    fprintf(stderr, "%.*s", static_cast<int>(data.size()), data.data());
    return Status::OK();
  }

  Status Close() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/util/dump_graph.cc", "Close");
 return Status::OK(); }

  Status Flush() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/util/dump_graph.cc", "Flush");

    fflush(stderr);
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_7(mht_7_v, 302, "", "./tensorflow/core/util/dump_graph.cc", "Name");

    *result = "stderr";
    return Status::OK();
  }

  Status Sync() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_8(mht_8_v, 310, "", "./tensorflow/core/util/dump_graph.cc", "Sync");
 return Status::OK(); }

  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_9(mht_9_v, 315, "", "./tensorflow/core/util/dump_graph.cc", "Tell");

    return errors::Unimplemented("Stream not seekable");
  }
};

Status CreateWritableFile(Env* env, const string& dirname, const string& name,
                          const string& suffix, string* filepath,
                          std::unique_ptr<WritableFile>* file) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("dirname: \"" + dirname + "\"");
   mht_10_v.push_back("name: \"" + name + "\"");
   mht_10_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_10(mht_10_v, 328, "", "./tensorflow/core/util/dump_graph.cc", "CreateWritableFile");

  string dir;
  if (!dirname.empty()) {
    dir = dirname;
  } else {
    const char* prefix = getenv("TF_DUMP_GRAPH_PREFIX");
    if (prefix != nullptr) dir = prefix;
  }
  if (dir.empty()) {
    LOG(WARNING)
        << "Failed to dump " << name << " because dump location is not "
        << " specified through either TF_DUMP_GRAPH_PREFIX environment "
        << "variable or function argument.";
    return errors::InvalidArgument("TF_DUMP_GRAPH_PREFIX not specified");
  }

  if (absl::EqualsIgnoreCase(dir, "sponge") ||
      absl::EqualsIgnoreCase(dir, "test_undeclared_outputs_dir")) {
    if (!io::GetTestUndeclaredOutputsDir(&dir)) {
      LOG(WARNING) << "TF_DUMP_GRAPH_PREFIX=sponge, but "
                      "TEST_UNDECLARED_OUTPUT_DIRS is not set, dumping to log";
      dir = "-";
    }
  }

  *filepath = "NULL";
  if (dir == "-") {
    *file = std::make_unique<StderrWritableFile>();
    *filepath = "(stderr)";
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  *filepath = io::JoinPath(dir, MakeUniqueFilename(name, suffix));
  return env->NewWritableFile(*filepath, file);
}

Status WriteTextProtoToUniqueFile(const tensorflow::protobuf::Message& proto,
                                  WritableFile* file) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_11(mht_11_v, 369, "", "./tensorflow/core/util/dump_graph.cc", "WriteTextProtoToUniqueFile");

  string s;
  if (!::tensorflow::protobuf::TextFormat::PrintToString(proto, &s)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  TF_RETURN_IF_ERROR(file->Append(s));
  StringPiece name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  return file->Close();
}

Status WriteTextProtoToUniqueFile(
    const tensorflow::protobuf::MessageLite& proto, WritableFile* file) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_12(mht_12_v, 386, "", "./tensorflow/core/util/dump_graph.cc", "WriteTextProtoToUniqueFile");

  string s;
  if (!SerializeToStringDeterministic(proto, &s)) {
    return errors::Internal("Failed to serialize proto to string.");
  }
  StringPiece name;
  TF_RETURN_IF_ERROR(file->Name(&name));
  VLOG(5) << name;
  VLOG(5) << s;
  TF_RETURN_IF_ERROR(file->Append(s));
  return file->Close();
}

}  // anonymous namespace

void SetGraphDumper(
    std::function<Status(const Graph& graph,
                         const FunctionLibraryDefinition* flib_def,
                         WritableFile*)>
        dumper,
    string suffix) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_13(mht_13_v, 410, "", "./tensorflow/core/util/dump_graph.cc", "SetGraphDumper");

  GraphDumperConfig& dumper_config = GetGraphDumperConfig();
  mutex_lock lock(dumper_config.mu);
  dumper_config.config.dumper = dumper;
  dumper_config.config.suffix = suffix;
}

string DumpGraphDefToFile(const string& name, GraphDef const& graph_def,
                          const string& dirname) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   mht_14_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_14(mht_14_v, 423, "", "./tensorflow/core/util/dump_graph.cc", "DumpGraphDefToFile");

  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(graph_def, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump Graph to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped Graph to " << filepath;
  return filepath;
}

string DumpCostGraphDefToFile(const string& name, CostGraphDef const& graph_def,
                              const string& dirname) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   mht_15_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_15(mht_15_v, 447, "", "./tensorflow/core/util/dump_graph.cc", "DumpCostGraphDefToFile");

  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(graph_def, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump Graph to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped Graph to " << filepath;
  return filepath;
}

string DumpGraphToFile(const string& name, Graph const& graph,
                       const FunctionLibraryDefinition* flib_def,
                       const string& dirname) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   mht_16_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_16(mht_16_v, 472, "", "./tensorflow/core/util/dump_graph.cc", "DumpGraphToFile");

  auto& dumper_config = GetGraphDumperConfig();
  if (dumper_config.IsSet()) {
    GraphDumperConfig::Config config;
    {
      mutex_lock lock(dumper_config.mu);
      config = dumper_config.config;
    }
    if (config.IsSet()) {
      string filepath;
      std::unique_ptr<WritableFile> file;
      Status status = CreateWritableFile(Env::Default(), dirname, name,
                                         config.suffix, &filepath, &file);
      if (!status.ok()) {
        return StrCat("(failed to create writable file: ", status.ToString(),
                      ")");
      }
      status = config.dumper(graph, flib_def, file.get());
      if (!status.ok()) {
        return StrCat("(failed to dump Graph to '", filepath,
                      "': ", status.ToString(), ")");
      }
      LOG(INFO) << "Dumped Graph to " << filepath;
      return filepath;
    }
  }

  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  if (flib_def) {
    *graph_def.mutable_library() = flib_def->ToProto();
  }
  return DumpGraphDefToFile(name, graph_def, dirname);
}

string DumpFunctionDefToFile(const string& name, FunctionDef const& fdef,
                             const string& dirname) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   mht_17_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdump_graphDTcc mht_17(mht_17_v, 513, "", "./tensorflow/core/util/dump_graph.cc", "DumpFunctionDefToFile");

  string filepath;
  std::unique_ptr<WritableFile> file;
  Status status = CreateWritableFile(Env::Default(), dirname, name, ".pbtxt",
                                     &filepath, &file);
  if (!status.ok()) {
    return StrCat("(failed to create writable file: ", status.ToString(), ")");
  }

  status = WriteTextProtoToUniqueFile(fdef, file.get());
  if (!status.ok()) {
    return StrCat("(failed to dump FunctionDef to '", filepath,
                  "': ", status.ToString(), ")");
  }
  LOG(INFO) << "Dumped FunctionDef to " << filepath;
  return filepath;
}

}  // namespace tensorflow
