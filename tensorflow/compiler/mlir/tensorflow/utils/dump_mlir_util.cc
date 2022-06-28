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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/platform/crash_analysis.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

using llvm::raw_ostream;

namespace tensorflow {
namespace {

struct NameCounts {
  mutex counts_mutex;
  llvm::StringMap<int64_t> counts;
};

std::string MakeUniqueFilename(string name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "MakeUniqueFilename");

  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0, e = name.size(); i < e; ++i) {
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

  std::string filename = name;
  if (count > 0) {
    filename = llvm::formatv("{0}_{1}", filename, count).str();
  }
  filename = llvm::Twine(filename).concat(".mlir").str();
  return filename;
}

// Simple raw_ostream that prints to stderr.
struct LogInfoRawStream : public llvm::raw_ostream {
  LogInfoRawStream() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_1(mht_1_v, 246, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "LogInfoRawStream");
 SetUnbuffered(); }
  ~LogInfoRawStream() override = default;
  uint64_t current_pos() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_2(mht_2_v, 251, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "current_pos");
 return 0; }

  void write_impl(const char* ptr, size_t size) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("ptr: \"" + (ptr == nullptr ? std::string("nullptr") : std::string((char*)ptr)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "write_impl");

    fprintf(stderr, "%.*s", static_cast<int>(size), ptr);
  }
};

// Simple raw_ostream that prints to a file.
struct WritableFileRawStream : public llvm::raw_ostream {
  explicit WritableFileRawStream(std::unique_ptr<WritableFile> file)
      : file(std::move(file)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_4(mht_4_v, 268, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "WritableFileRawStream");

    SetUnbuffered();
  }
  ~WritableFileRawStream() override = default;
  uint64_t current_pos() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_5(mht_5_v, 275, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "current_pos");
 return 0; }

  void write_impl(const char* ptr, size_t size) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("ptr: \"" + (ptr == nullptr ? std::string("nullptr") : std::string((char*)ptr)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_6(mht_6_v, 281, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "write_impl");

    // Write the file if it is still valid. If the write fails, null out the
    // file to avoid encountering another error.
    if (file && !file->Append(StringPiece(ptr, size)).ok()) {
      file = nullptr;
    }
  }

  // The file being written to.
  std::unique_ptr<WritableFile> file;
};

struct CrashReproducerStream : public mlir::PassManager::ReproducerStream {
  CrashReproducerStream(llvm::StringRef name,
                        std::unique_ptr<llvm::raw_ostream> file)
      : name(name), ostream(std::move(file)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_7(mht_7_v, 299, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "CrashReproducerStream");
}

  llvm::StringRef description() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_8(mht_8_v, 304, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "description");
 return name; }
  raw_ostream& os() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_9(mht_9_v, 308, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "os");
 return *ostream; }

 private:
  std::string name;
  std::unique_ptr<llvm::raw_ostream> ostream;
};

// MLIR crash reproducer which reports failures to the crash analysis system.
struct CrashAnalysisCrashReproducerStream
    : public mlir::PassManager::ReproducerStream {
 public:
  CrashAnalysisCrashReproducerStream()
      : internal_str(""), string_stream(internal_str) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_10(mht_10_v, 323, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "CrashAnalysisCrashReproducerStream");
}

  ~CrashAnalysisCrashReproducerStream() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_11(mht_11_v, 328, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "~CrashAnalysisCrashReproducerStream");

    crash_analysis::ReportEvent(
        "mlir_crash_reproducer.mlir",
        "Pass pipeline failure; crash reproducer attached",
        string_stream.str());
  }

  llvm::StringRef description() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_12(mht_12_v, 338, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "description");
 return "mlir_crash_reproducer"; }
  raw_ostream& os() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_13(mht_13_v, 342, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "os");
 return string_stream; }

 private:
  std::string internal_str;
  llvm::raw_string_ostream string_stream;
};

}  // namespace

const char kCrashReproducerStdErr[] = "-";
const char kCrashReproducerCrashAnalysis[] = "crash_analysis";

Status CreateFileForDumping(llvm::StringRef name,
                            std::unique_ptr<raw_ostream>* os,
                            std::string* filepath, llvm::StringRef dirname) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_14(mht_14_v, 359, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "CreateFileForDumping");

  std::string dir;
  if (!dirname.empty())
    dir = std::string(dirname);
  else
    dir = GetDumpDirFromEnvVar();

  if (dir.empty()) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "(TF_DUMP_GRAPH_PREFIX not specified)");
  }

  if (dir == kCrashReproducerStdErr) {
    *os = std::make_unique<LogInfoRawStream>();
    *filepath = "(stderr)";
    return Status();
  }

  // Get a valid file path to dump with.
  Env* env = Env::Default();
  Status status = env->RecursivelyCreateDir(dir);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create '" << dir
                 << "' directory for dumping: " << status;
    return Status(error::Code::UNAVAILABLE, "(unavailable)");
  }
  *filepath = io::JoinPath(dir, MakeUniqueFilename(std::string(name)));

  // Try to open the file and generate a raw_ostream.
  std::unique_ptr<WritableFile> file;
  status = env->NewWritableFile(*filepath, &file);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create file '" << filepath << "': " << status;
    return Status(error::Code::UNAVAILABLE, "(unavailable)");
  }
  *os = std::make_unique<WritableFileRawStream>(std::move(file));
  return Status();
}

std::string DumpCrashReproducerToFile(llvm::StringRef name,
                                      const mlir::PassManager& pm,
                                      mlir::Operation* op,
                                      llvm::StringRef dirname) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_15(mht_15_v, 404, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "DumpCrashReproducerToFile");

  std::unique_ptr<llvm::raw_ostream> os;
  std::string filepath;
  Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return result.error_message();

  std::string str;
  llvm::raw_string_ostream passOS(str);
  llvm::interleaveComma(pm.getPasses(), passOS, [&](mlir::Pass& pass) {
    pass.printAsTextualPipeline(passOS);
  });
  *os << "// configuration: -pass-pipeline='" << passOS.str() << "'";
  if (op->getContext()->isMultithreadingEnabled())
    *os << " -mlir-disable-threading";
  *os << " -verify-each";
  *os << "\n";
  op->print(*os, mlir::OpPrintingFlags().useLocalScope().printGenericOpForm());
  LOG(INFO) << "Dumped MLIR operation '" << op->getName().getStringRef().str()
            << "' to '" << filepath << "'";
  return filepath;
}

std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_16(mht_16_v, 430, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "DumpMlirOpToFile");

  std::unique_ptr<raw_ostream> os;
  std::string filepath;
  Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return result.error_message();

  op->print(*os, mlir::OpPrintingFlags().useLocalScope().printGenericOpForm());
  LOG(INFO) << "Dumped MLIR operation '" << op->getName().getStringRef().str()
            << "' to '" << filepath << "'";
  return filepath;
}

std::string GetDumpDirFromEnvVar() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_17(mht_17_v, 445, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "GetDumpDirFromEnvVar");

  const char* prefix_env = getenv("TF_DUMP_GRAPH_PREFIX");
  if (!prefix_env) {
    LOG(WARNING)
        << "Failed to dump MLIR module because dump location is not "
        << "specified through TF_DUMP_GRAPH_PREFIX environment variable.";
    return "";
  }

  std::string result = prefix_env;

  if (absl::EqualsIgnoreCase(result, "sponge") &&
      !io::GetTestUndeclaredOutputsDir(&result)) {
    LOG(WARNING) << "TF_DUMP_GRAPH_PREFIX=sponge but "
                    "TEST_UNDECLARED_OUTPUT_DIRS is not set";
    return "";
  }
  return result;
}

std::string DumpRawStringToFile(llvm::StringRef name, llvm::StringRef content,
                                llvm::StringRef dirname) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_18(mht_18_v, 469, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "DumpRawStringToFile");

  std::unique_ptr<raw_ostream> os;
  std::string filepath;
  Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return result.error_message();

  (*os) << content;
  LOG(INFO) << "Outputted requested string to '" << filepath << "'";
  return filepath;
}

void SetCrashReproducer(mlir::PassManager& pm, llvm::StringRef dir_path) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_19(mht_19_v, 483, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "SetCrashReproducer");

  std::string path = dir_path.str();
  if (path.empty() || path == kCrashReproducerCrashAnalysis) {
    if (getenv("MLIR_CRASH_REPRODUCER_DIRECTORY"))
      path = getenv("MLIR_CRASH_REPRODUCER_DIRECTORY");
    else if (getenv("TEST_UNDECLARED_OUTPUTS_DIR"))
      path = "sponge";
  }
  if (path.empty()) {
    LOG_FIRST_N(INFO, 1) << "disabling MLIR crash reproducer, set env var "
                            "`MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.";
    return;
  }

  // Output dirs "sponge" (case-insensitive) have a special meaning: Dump into
  // the directory specified by the environment variable
  // TEST_UNDECLARED_OUTPUTS_DIR.
  string lower_path = absl::AsciiStrToLower(path);
  if (lower_path == "sponge") {
    if (!tensorflow::io::GetTestUndeclaredOutputsDir(&path)) {
      LOG(ERROR) << "MLIR crash reproducer is set to '" << dir_path.str()
                 << "', but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                    "is not set, so cannot dump anywhere.";
      return;
    }
  }

  // kCrashReproducerStdErr and kCrashReproducerCrashAnalysis settings do not
  // require explicit file creation.
  if (path != kCrashReproducerStdErr && path != kCrashReproducerCrashAnalysis) {
    auto* env = tensorflow::Env::Default();
    auto status = env->RecursivelyCreateDir(path);
    if (!status.ok()) {
      LOG(WARNING) << "cannot create directory '" + path +
                          "': " + status.error_message();
      return;
    }

    path += "/mlir_reproducer_";

    if (!tensorflow::Env::Default()->CreateUniqueFileName(&path, ".mlir")) {
      LOG(WARNING) << "cannot create unique filename, won't enable MLIR crash "
                      "reproducer.";
      return;
    }
  }

  mlir::PassManager::ReproducerStreamFactory factory =
      [path](std::string& error)
      -> std::unique_ptr<mlir::PassManager::ReproducerStream> {
    if (path == kCrashReproducerStdErr)
      return std::make_unique<CrashReproducerStream>(
          "(stderr)", std::make_unique<LogInfoRawStream>());
    if (path == kCrashReproducerCrashAnalysis) {
      return std::make_unique<CrashAnalysisCrashReproducerStream>();
    }

    // Try to open the file and generate a raw_ostream.
    std::unique_ptr<WritableFile> file;
    Status status = tensorflow::Env::Default()->NewWritableFile(path, &file);
    if (!status.ok()) {
      error = absl::StrCat("Failed to create file '", path,
                           "': ", status.error_message());
      return nullptr;
    }
    return std::make_unique<CrashReproducerStream>(
        path, std::make_unique<WritableFileRawStream>(std::move(file)));
  };
  pm.enableCrashReproducerGeneration(factory, /*genLocalReproducer=*/false);
}

void applyTensorflowAndCLOptions(mlir::PassManager& pm,
                                 llvm::StringRef dir_path) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSutilsPSdump_mlir_utilDTcc mht_20(mht_20_v, 558, "", "./tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc", "applyTensorflowAndCLOptions");

  mlir::applyPassManagerCLOptions(pm);
  SetCrashReproducer(pm, dir_path);
}

}  // namespace tensorflow
