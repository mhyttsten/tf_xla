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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc() {
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

#include "tensorflow/compiler/xla/service/dump.h"

#include <memory>
#include <queue>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/LocationSnapshot.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

namespace {

using absl::StrCat;
using absl::StrFormat;
using absl::string_view;

struct CanonicalDebugOptions {
  explicit CanonicalDebugOptions(const DebugOptions& opts)
      : dump_to(opts.xla_dump_to()),
        dump_as_text(opts.xla_dump_hlo_as_text()),
        dump_as_proto(opts.xla_dump_hlo_as_proto()),
        dump_as_dot(opts.xla_dump_hlo_as_dot()),
        dump_as_html(opts.xla_dump_hlo_as_html()),
        dump_as_url(opts.xla_dump_hlo_as_url()),
        dump_fusion_visualization(opts.xla_dump_fusion_visualization()),
        dump_snapshots(opts.xla_dump_hlo_snapshots()),
        dump_include_timestamp(opts.xla_dump_include_timestamp()),
        dump_max_hlo_modules(opts.xla_dump_max_hlo_modules()),
        dump_module_metadata(opts.xla_dump_module_metadata()),
        dump_compress_protos(opts.xla_dump_compress_protos()),
        dump_hlo_metadata(!opts.xla_dump_disable_metadata()),
        dump_as_long_text(opts.xla_dump_hlo_as_long_text()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_0(mht_0_v, 233, "", "./tensorflow/compiler/xla/service/dump.cc", "CanonicalDebugOptions");

    // This constructor examines the values in `opts` and turns on other flags
    // based on what we think is the user's intent.  To reduce confusion about
    // what was a user-specified value versus an extrapolated value, within this
    // function we treat this struct's members as write-only, and read only from
    // `opts`.

    // Did the user specify an explicit format for dumping?
    bool output_format_other_than_url_specified =
        opts.xla_dump_hlo_as_text() || opts.xla_dump_hlo_as_proto() ||
        opts.xla_dump_hlo_as_dot() || opts.xla_dump_hlo_as_html() ||
        opts.xla_dump_hlo_snapshots();
    bool output_format_specified =
        output_format_other_than_url_specified || opts.xla_dump_hlo_as_url();

    // If we haven't specified an output format, default to dumping as text.
    if (!output_format_specified) {
      dump_as_text = true;
    }

    // Disable dumping if specified by the user.
    if (!opts.xla_detailed_logging_and_dumping()) {
      dump_to = "";
    }

    // If dump_to is empty, default to dumping to stdout, so long as some dump
    // format other than dump-as-url was specified.  If the user only specified
    // --xla_dump_hlo_as_url, then don't dump to stdout, that is likely noise
    // they don't want.
    if (opts.xla_dump_to().empty() && output_format_other_than_url_specified) {
      dump_to = "-";
    }

    // If we specified a regular expression restricting which modules to dump,
    // respect that.
    //
    // If we didn't specify which modules to dump but we passed some other flag
    // which implies dumping modules, dump all modules.
    //
    // Otherwise, don't dump any HLO modules.
    if (!opts.xla_dump_hlo_module_re().empty()) {
      // RE2 object is not copyable, and we can't capture "by move", so we
      // resort to this hack.
      std::string pattern = opts.xla_dump_hlo_module_re();
      should_dump_module = [pattern](string_view module_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("module_name: \"" + std::string(module_name.data(), module_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_1(mht_1_v, 281, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");

        return RE2::PartialMatch(module_name, pattern);
      };
    } else if (!opts.xla_dump_hlo_pass_re().empty() ||
               !opts.xla_dump_to().empty() || output_format_specified) {
      should_dump_module = [](string_view) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_2(mht_2_v, 289, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return true; };
    } else {
      should_dump_module = [](string_view) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_3(mht_3_v, 294, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return false; };
    }

    // Initialize should_dump_pass.  This one is easy: We only dump per-pass
    // data if the user asked for it explicitly.
    if (!opts.xla_dump_hlo_pass_re().empty()) {
      std::string pattern = opts.xla_dump_hlo_pass_re();
      should_dump_pass = [pattern](string_view pass_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_4(mht_4_v, 305, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");

        return RE2::PartialMatch(pass_name, pattern);
      };
    } else {
      should_dump_pass = [](string_view) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_5(mht_5_v, 312, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return false; };
    }

    // Initialize should_dump_pipeline. If the option was not specified, dump
    // all pipelines. Otherwise dump only those pipelines that user asked for
    // explicitly.
    if (!opts.xla_dump_hlo_pipeline_re().empty()) {
      std::string pattern = opts.xla_dump_hlo_pipeline_re();
      should_dump_pipeline = [pattern](string_view pipeline_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("pipeline_name: \"" + std::string(pipeline_name.data(), pipeline_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_6(mht_6_v, 324, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");

        return RE2::PartialMatch(pipeline_name, pattern);
      };
    } else {
      should_dump_pipeline = [](string_view) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_7(mht_7_v, 331, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return true; };
    }

    // Output dirs "sponge" and "test_undeclared_outputs_dir" (case-insensitive)
    // have a special meaning: Dump into the directory specified by the
    // environment variable TEST_UNDECLARED_OUTPUTS_DIR.
    std::string dump_to_lower = absl::AsciiStrToLower(dump_to);
    if (dump_to_lower == "sponge" ||
        dump_to_lower == "test_undeclared_outputs_dir") {
      if (!tensorflow::io::GetTestUndeclaredOutputsDir(&dump_to)) {
        LOG(ERROR) << "--xla_dump_to=" << opts.xla_dump_to()
                   << ", but environment variable TEST_UNDECLARED_OUTPUTS_DIR "
                      "is not set, so cannot dump anywhere.";
        should_dump_module = [](string_view) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_8(mht_8_v, 347, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return false; };
        should_dump_pass = [](string_view) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_9(mht_9_v, 351, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return false; };
        should_dump_pipeline = [](string_view) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_10(mht_10_v, 355, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");
 return false; };
      }
    }
  }

  bool dumping_to_stdout() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_11(mht_11_v, 363, "", "./tensorflow/compiler/xla/service/dump.cc", "dumping_to_stdout");
 return dump_to == "-"; }

  std::string dump_to;
  std::function<bool(string_view module_name)> should_dump_module;
  std::function<bool(string_view pass_name)> should_dump_pass;
  std::function<bool(string_view pipeline_name)> should_dump_pipeline;

  // dump_ir isn't present here because this file is mostly concerned with
  // dumping HLO.
  bool dump_as_text;
  bool dump_as_proto;
  bool dump_as_dot;
  bool dump_as_html;
  bool dump_as_url;
  bool dump_fusion_visualization;
  bool dump_snapshots;
  bool dump_include_timestamp;
  int64_t dump_max_hlo_modules;
  bool dump_module_metadata;
  bool dump_compress_protos;
  bool dump_hlo_metadata;
  bool dump_as_long_text;
};

// Helper class to hold a list of functions that produces data to be written to
// a file in multiple stages, so that we can lower the peak memory usage.
// Ideally we should migrate this whole file to use an I/O stream style API.
class DataProducer {
 public:
  void Append(std::function<std::string()> produce_func) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_12(mht_12_v, 395, "", "./tensorflow/compiler/xla/service/dump.cc", "Append");

    produce_funcs_.push(std::move(produce_func));
  }

  std::function<std::string()> Next() {
    if (produce_funcs_.empty()) {
      return nullptr;
    }
    auto next = std::move(produce_funcs_.front());
    produce_funcs_.pop();
    return next;
  }

 private:
  std::queue<std::function<std::string()>> produce_funcs_;
};

static Status WriteStringToFile(tensorflow::Env* env, const std::string& fname,
                                DataProducer& data_producer, bool compressed) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_13(mht_13_v, 417, "", "./tensorflow/compiler/xla/service/dump.cc", "WriteStringToFile");

  std::unique_ptr<tensorflow::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  if (compressed) {
    auto gz_opts = tensorflow::io::ZlibCompressionOptions::GZIP();
    tensorflow::io::ZlibOutputBuffer gz_file(
        file.get(), gz_opts.input_buffer_size, gz_opts.output_buffer_size,
        gz_opts);
    TF_RETURN_IF_ERROR(gz_file.Init());
    while (auto next_producer = data_producer.Next()) {
      TF_RETURN_IF_ERROR(gz_file.Append(next_producer()));
    }
    return gz_file.Close();
  } else {
    while (auto next_producer = data_producer.Next()) {
      TF_RETURN_IF_ERROR(file->Append(next_producer()));
    }
    return file->Close();
  }
}

static Status WriteStringToFile(tensorflow::Env* env, const std::string& fname,
                                absl::string_view data, bool compressed) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("fname: \"" + fname + "\"");
   mht_14_v.push_back("data: \"" + std::string(data.data(), data.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_14(mht_14_v, 444, "", "./tensorflow/compiler/xla/service/dump.cc", "WriteStringToFile");

  if (!compressed) {
    return tensorflow::WriteStringToFile(env, fname, data);
  }
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(fname, &file));
  auto gz_opts = tensorflow::io::ZlibCompressionOptions::GZIP();
  tensorflow::io::ZlibOutputBuffer gz_file(file.get(),
                                           gz_opts.input_buffer_size,
                                           gz_opts.output_buffer_size, gz_opts);
  TF_RETURN_IF_ERROR(gz_file.Init());
  TF_RETURN_IF_ERROR(gz_file.Append(data));
  return gz_file.Close();
}

static absl::optional<std::string> GetDumpFilePath(
    string_view filename, const CanonicalDebugOptions& opts) {
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return absl::nullopt;
  }

  if (opts.dump_to.empty()) {
    return absl::nullopt;
  }

  const std::string& dir = opts.dump_to;
  VLOG(1) << "Dumping " << filename << " to " << dir;

  tensorflow::Env* env = tensorflow::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return absl::nullopt;
    }
  }

  // Make sure we are not going to dump more modules than the user has asked.
  if (opts.dump_max_hlo_modules > 0) {
    std::vector<std::string> matches;
    auto pattern = tensorflow::io::JoinPath(dir, "*module_*.*");
    auto status = env->GetMatchingPaths(pattern, &matches);
    if (!status.ok()) {
      LOG(ERROR) << "Could not get matching paths for pattern " << pattern
                 << ": " << status;
    }
    static const LazyRE2 module_id_regex = {R"(.*module_(\d+)\..*)"};
    absl::flat_hash_set<int64_t> dumped_module_ids;
    for (const std::string& match : matches) {
      int64_t dumped_module_id;
      if (RE2::FullMatch(match, *module_id_regex, &dumped_module_id)) {
        dumped_module_ids.insert(dumped_module_id);
      }
    }
    if (dumped_module_ids.size() >= opts.dump_max_hlo_modules) {
      int64_t module_id;
      if (RE2::FullMatch(filename, *module_id_regex, &module_id) &&
          !dumped_module_ids.contains(module_id)) {
        LOG(ERROR) << "Have already dumped " << dumped_module_ids.size()
                   << " modules, more than the limit of "
                   << opts.dump_max_hlo_modules;
        return absl::nullopt;
      }
    }
  }

  return tensorflow::io::JoinPath(dir, SanitizeFileName(std::string(filename)));
}

static absl::optional<std::string> DumpToFileInDirImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts, bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) return absl::nullopt;

  auto status = WriteStringToFile(tensorflow::Env::Default(), *file_path,
                                  contents, compress);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return absl::nullopt;
  }

  return file_path;
}

static absl::optional<std::string> DumpToFileInDirImpl(
    string_view filename, DataProducer& data_producer,
    const CanonicalDebugOptions& opts, bool compress = false) {
  auto file_path = GetDumpFilePath(filename, opts);
  if (!file_path) return absl::nullopt;

  auto status = WriteStringToFile(tensorflow::Env::Default(), *file_path,
                                  data_producer, compress);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << *file_path << ": "
               << status;
    return absl::nullopt;
  }

  return file_path;
}

static absl::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, string_view contents,
    const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    std::cout << "*** Begin " << filename << " ***\n"
              << contents << "\n*** End " << filename << " ***" << std::endl;
    return absl::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, contents, opts);
}

static absl::optional<std::string> DumpToFileInDirOrStdoutImpl(
    string_view filename, DataProducer& data_producer,
    const CanonicalDebugOptions& opts) {
  // Dump to stdout if that's called for.
  if (opts.dumping_to_stdout()) {
    std::cout << "*** Begin " << filename << " ***\n";
    while (auto next_producer = data_producer.Next()) {
      std::cout << next_producer();
    }
    std::cout << "\n*** End " << filename << " ***" << std::endl;
    return absl::nullopt;
  }

  // Otherwise, dump to a file.
  return DumpToFileInDirImpl(filename, data_producer, opts);
}

// Returns whether the computation is trivial enough not to warrant dumping.
// Currently skips instructions where the root instruction has only parameters
// as operands AND is not a fusion.
static bool IsTrivial(const HloComputation& computation) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_15(mht_15_v, 591, "", "./tensorflow/compiler/xla/service/dump.cc", "IsTrivial");

  const HloInstruction* root = computation.root_instruction();
  return absl::c_all_of(root->operands(),
                        [&](const HloInstruction* op) {
                          return op->opcode() == HloOpcode::kParameter;
                        }) &&
         root->opcode() != HloOpcode::kFusion;
}

// Returns full file paths of all dumps of the module.
static std::vector<std::string> DumpHloModuleImpl(
    const HloModule& module, const BufferAssignment* buffer_assn,
    const HloExecutionProfile* profile, string_view prefix, string_view suffix,
    const CanonicalDebugOptions& opts) {
  std::string filename = FilenameFor(module, prefix, suffix);

  std::vector<absl::optional<std::string>> file_paths;

  if (opts.dump_as_text) {
    auto print_options = opts.dump_as_long_text
                             ? HloPrintOptions()
                             : HloPrintOptions::ShortParsable();
    print_options.set_print_large_constants(false);
    print_options.set_print_control_dependencies(true);
    print_options.set_print_operand_index_annotation_interval(5);
    print_options.set_print_backend_config(true);
    print_options.set_print_metadata(opts.dump_hlo_metadata);
    file_paths.push_back(DumpToFileInDirOrStdoutImpl(
        StrCat(filename, ".txt"), module.ToString(print_options), opts));
    if (buffer_assn) {
      DataProducer data_producer;
      data_producer.Append([&] { return buffer_assn->ToString(); });
      data_producer.Append([&] { return "\n\n"; });
      data_producer.Append(
          [&] { return buffer_assn->hlo_live_range().ToString(); });
      file_paths.push_back(DumpToFileInDirOrStdoutImpl(
          StrCat(filename, "-buffer-assignment.txt"), data_producer, opts));
    }
  }

  if (opts.dump_as_proto) {
    HloProto module_proto =
        buffer_assn ? MakeHloProto(module, *buffer_assn) : MakeHloProto(module);
    std::string pb;
    if (!tensorflow::SerializeToStringDeterministic(module_proto, &pb)) {
      pb = "Failed to serialize HLO module proto.";
    }
    file_paths.push_back(DumpToFileInDirImpl(
        StrCat(filename, opts.dump_compress_protos ? ".hlo.pb.gz" : ".hlo.pb"),
        pb, opts, opts.dump_compress_protos));
  }

  auto render_graph = [&](RenderedGraphFormat format) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_16(mht_16_v, 646, "", "./tensorflow/compiler/xla/service/dump.cc", "lambda");

    StatusOr<std::string> rendered_graph = RenderGraph(
        *module.entry_computation(),
        /*label=*/filename, module.config().debug_options(), format, profile);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return StrFormat("Error rendering graph: %s",
                     rendered_graph.status().ToString());
  };

  if (opts.dump_as_dot) {
    file_paths.push_back(
        DumpToFileInDirImpl(StrFormat("%s.dot", filename),
                            render_graph(RenderedGraphFormat::kDot), opts));
  }

  if (opts.dump_as_html) {
    file_paths.push_back(
        DumpToFileInDirImpl(StrFormat("%s.html", filename),
                            render_graph(RenderedGraphFormat::kHtml), opts));
  }

  if (opts.dump_fusion_visualization) {
    for (const HloComputation* computation :
         module.MakeNonfusionComputations()) {
      if (IsTrivial(*computation)) {
        VLOG(1) << "Skipping computation " << computation->name()
                << " as trivial";
        continue;
      }

      StatusOr<std::string> rendered_graph = WrapFusionExplorer(*computation);
      if (!rendered_graph.ok()) {
        VLOG(1) << "Skipping fusion visualization"
                << " for computation " << computation->name()
                << " due to: " << rendered_graph.status().ToString();
        continue;
      }
      file_paths.push_back(DumpToFileInDirImpl(
          FilenameFor(module, computation->name(), "_fusion.html"),
          *rendered_graph, opts));
    }
  }

  // Special case for rendering graphs as URLs.  We'll dump them to a file
  // because why not, but we always log them to stdout as well.
  if (opts.dump_as_url) {
    std::string url = render_graph(RenderedGraphFormat::kUrl);
    std::cout << filename << " --> " << url << std::endl;
    if (!opts.dumping_to_stdout()) {
      file_paths.push_back(
          DumpToFileInDirImpl(StrFormat("%s.url", filename), url, opts));
    }
  }

  std::vector<std::string> dumped_file_paths;
  for (const absl::optional<std::string>& path : file_paths) {
    if (path.has_value()) {
      dumped_file_paths.push_back(*path);
    }
  }
  return dumped_file_paths;
}

static void DumpHloModuleMetadata(
    const HloModuleMetadataProto& metadata, const CanonicalDebugOptions& opts,
    absl::flat_hash_set<int64_t>* dumped_module_ids) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_17(mht_17_v, 716, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleMetadata");

  // Return if metadata for this module has already been dumped.
  if (!dumped_module_ids->insert(metadata.canonical_module_id()).second) {
    return;
  }
  std::string filename = absl::StrFormat("module_%04d.metadata.textproto",
                                         metadata.canonical_module_id());
  std::string content;
  if (tensorflow::protobuf::TextFormat::PrintToString(metadata, &content)) {
    DumpToFileInDirImpl(filename, content, opts);
  } else {
    LOG(ERROR) << "Failed to convert HloModuleMetadataProto to text.";
  }
}

static absl::Mutex mu(absl::kConstInit);

// Maps a module's unique ID to a counter indicating how many times we've dumped
// this module during the compilation pipeline.  This lets us keep the filenames
// ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_step_number ABSL_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64_t, int64_t>();

// Maps a module's unique ID to a timestamp indicating when we've first dumped
// this module during the compilation pipeline and when we first started
// compiling this module.  This lets us keep the filenames ordered nicely.
//
// Entries added here leak forever; we have no way to GC them when a module
// dies.  But we only add an entry if dumping is enabled for this module, and
// dumping a module leaks buffer space in stdout or bytes on disk *way* faster
// than this hashtable leaks memory.
static auto& module_id_to_timestamp ABSL_GUARDED_BY(mu) =
    *new absl::flat_hash_map<int64_t, uint64_t>();

int64_t StepNumberForModule(const HloModule& module) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_18(mht_18_v, 758, "", "./tensorflow/compiler/xla/service/dump.cc", "StepNumberForModule");

  absl::MutexLock lock(&mu);
  return module_id_to_step_number[module.unique_id()]++;
}

}  // namespace

// Get a timestamp which we can use as a filename prefix specific to this
// module.
std::string TimestampFor(const HloModule& module) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_19(mht_19_v, 770, "", "./tensorflow/compiler/xla/service/dump.cc", "TimestampFor");

  if (!module.config().debug_options().xla_dump_include_timestamp()) {
    return "";
  }
  absl::MutexLock lock(&mu);
  auto timestamp_emplace = module_id_to_timestamp.try_emplace(
      module.unique_id(), tensorflow::Env::Default()->NowMicros());
  return std::to_string(timestamp_emplace.first->second);
}

static std::string FilenameFor(int unique_id, string_view module_name,
                               string_view prefix, string_view suffix) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("module_name: \"" + std::string(module_name.data(), module_name.size()) + "\"");
   mht_20_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   mht_20_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_20(mht_20_v, 787, "", "./tensorflow/compiler/xla/service/dump.cc", "FilenameFor");

  std::string filename;
  if (!prefix.empty()) {
    absl::StrAppend(&filename, prefix, ".");
  }
  absl::StrAppendFormat(&filename, "module_%04d", unique_id);
  if (!module_name.empty()) {
    absl::StrAppend(&filename, ".", module_name);
  }
  absl::StrAppend(&filename, ".", suffix);
  // Skip the module name if the resulting length is too long.
  if (!module_name.empty() && filename.size() > 255) {
    return FilenameFor(unique_id, "", prefix, suffix);
  }
  return filename;
}

std::string FilenameFor(const HloModule& module, string_view prefix,
                        string_view suffix) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("prefix: \"" + std::string(prefix.data(), prefix.size()) + "\"");
   mht_21_v.push_back("suffix: \"" + std::string(suffix.data(), suffix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_21(mht_21_v, 810, "", "./tensorflow/compiler/xla/service/dump.cc", "FilenameFor");

  return FilenameFor(module.unique_id(), module.name(), prefix, suffix);
}

void DumpToFileInDir(const HloModule& module, string_view file_prefix,
                     string_view file_suffix, string_view contents) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("file_prefix: \"" + std::string(file_prefix.data(), file_prefix.size()) + "\"");
   mht_22_v.push_back("file_suffix: \"" + std::string(file_suffix.data(), file_suffix.size()) + "\"");
   mht_22_v.push_back("contents: \"" + std::string(contents.data(), contents.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_22(mht_22_v, 821, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpToFileInDir");

  DumpToFileInDir(module.config().debug_options(),
                  FilenameFor(module, file_prefix, file_suffix), contents);
}

void DumpToFileInDir(const DebugOptions& debug_options,
                     absl::string_view filename, absl::string_view contents) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   mht_23_v.push_back("contents: \"" + std::string(contents.data(), contents.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_23(mht_23_v, 832, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpToFileInDir");

  DumpToFileInDirImpl(filename, contents, CanonicalDebugOptions(debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("file_prefix: \"" + std::string(file_prefix.data(), file_prefix.size()) + "\"");
   mht_24_v.push_back("file_suffix: \"" + std::string(file_suffix.data(), file_suffix.size()) + "\"");
   mht_24_v.push_back("contents: \"" + std::string(contents.data(), contents.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_24(mht_24_v, 843, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpToFileInDirOrStdout");

  DumpToFileInDirOrStdoutImpl(
      FilenameFor(module, file_prefix, file_suffix), contents,
      CanonicalDebugOptions(module.config().debug_options()));
}

void DumpToFileInDirOrStdout(const DebugOptions& debug_options, int unique_id,
                             string_view module_name, string_view file_prefix,
                             string_view file_suffix, string_view contents) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("module_name: \"" + std::string(module_name.data(), module_name.size()) + "\"");
   mht_25_v.push_back("file_prefix: \"" + std::string(file_prefix.data(), file_prefix.size()) + "\"");
   mht_25_v.push_back("file_suffix: \"" + std::string(file_suffix.data(), file_suffix.size()) + "\"");
   mht_25_v.push_back("contents: \"" + std::string(contents.data(), contents.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_25(mht_25_v, 858, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpToFileInDirOrStdout");

  DumpToFileInDirOrStdoutImpl(
      FilenameFor(unique_id, module_name, file_prefix, file_suffix), contents,
      CanonicalDebugOptions(debug_options));
}

void DumpToFileInDirOrStdout(const HloModule& module, string_view file_prefix,
                             mlir::Operation* op) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("file_prefix: \"" + std::string(file_prefix.data(), file_prefix.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_26(mht_26_v, 869, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpToFileInDirOrStdout");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.dumping_to_stdout()) return op->dump();

  auto file_path =
      GetDumpFilePath(FilenameFor(module, file_prefix, "mlir"), opts);
  if (!file_path) return;

  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(llvm::SmallString<32>(*file_path), &error);
  if (!outputFile) {
    LOG(ERROR) << "Error: " << error << std::endl
               << "Failed to open file: " << *file_path;
    return;
  }

  op->print(outputFile->os(), mlir::OpPrintingFlags().useLocalScope());
  outputFile->keep();
}

void DumpProtobufToFile(const tensorflow::protobuf::Message& proto,
                        const DebugOptions& debug_options,
                        absl::string_view filename) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("filename: \"" + std::string(filename.data(), filename.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_27(mht_27_v, 896, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpProtobufToFile");

  CanonicalDebugOptions opts(debug_options);
  tensorflow::Env* env = tensorflow::Env::Default();
  const std::string& dir = opts.dump_to;
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA execution options: " << status;
      return;
    }
  }
  if (env->IsDirectory(dir).ok()) {
    const std::string path = tensorflow::io::JoinPath(dir, filename);
    Status status;
    if (opts.dump_as_text) {
      status =
          tensorflow::WriteTextProto(env, absl::StrCat(path, ".txt"), proto);
    } else {
      status =
          tensorflow::WriteBinaryProto(env, absl::StrCat(path, ".pb"), proto);
    }
    if (!status.ok()) {
      LOG(ERROR) << "Could not write XLA debug data to " << filename << ": "
                 << status;
    }
  }
}

void DumpPerModuleProtobufToFile(const HloModule& module,
                                 const tensorflow::protobuf::Message& proto,
                                 const DebugOptions& debug_options,
                                 absl::string_view name) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_28(mht_28_v, 932, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpPerModuleProtobufToFile");

  const std::string filename = FilenameFor(module, TimestampFor(module), name);
  DumpProtobufToFile(proto, debug_options, filename);
}

void DumpHloModuleIfEnabled(const HloModule& module, string_view name) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_29(mht_29_v, 941, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleIfEnabled");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                      TimestampFor(module), name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            string_view name) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_30(mht_30_v, 955, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleIfEnabled");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, &buffer_assn, /*profile=*/nullptr,
                      TimestampFor(module), name, opts);
  }
}

void DumpHloModuleIfEnabled(const HloModule& module,
                            const HloExecutionProfile& profile,
                            string_view name) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_31(mht_31_v, 969, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleIfEnabled");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (opts.should_dump_module(module.name())) {
    DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, &profile,
                      TimestampFor(module), name, opts);
  }
}

bool DumpingEnabledForHloModule(string_view hlo_module_name,
                                const DebugOptions& opts) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("hlo_module_name: \"" + std::string(hlo_module_name.data(), hlo_module_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_32(mht_32_v, 982, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpingEnabledForHloModule");

  return CanonicalDebugOptions(opts).should_dump_module(hlo_module_name);
}

bool DumpingToStdout(const DebugOptions& opts) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_33(mht_33_v, 989, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpingToStdout");

  return CanonicalDebugOptions(opts).dumping_to_stdout();
}

std::vector<std::string> DumpHloModuleBetweenPassesIfEnabled(
    string_view pipeline_name, string_view before_pass_name,
    string_view after_pass_name, const HloModule& module) {
  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name())) {
    return {};
  }

  if (!opts.should_dump_pass(before_pass_name) &&
      !opts.should_dump_pass(after_pass_name)) {
    return {};
  }

  if (!opts.should_dump_pipeline(pipeline_name)) {
    return {};
  }

  int64_t step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  std::string filename_suffix =
      StrFormat("%04d.%s.after_%s.before_%s", step_number, pipeline_name,
                after_pass_name, before_pass_name);
  return DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                           timestamp, filename_suffix, opts);
}

void DumpHloModuleDuringPassIfEnabled(string_view pass_name,
                                      string_view step_name,
                                      const HloModule& module) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("pass_name: \"" + std::string(pass_name.data(), pass_name.size()) + "\"");
   mht_34_v.push_back("step_name: \"" + std::string(step_name.data(), step_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_34(mht_34_v, 1027, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleDuringPassIfEnabled");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) ||
      !opts.should_dump_pass(pass_name)) {
    return;
  }

  int64_t step_number = StepNumberForModule(module);
  std::string timestamp = TimestampFor(module);

  std::string filename_suffix =
      StrFormat("%04d.%s.%s", step_number, pass_name, step_name);
  DumpHloModuleImpl(module, /*buffer_assn=*/nullptr, /*profile=*/nullptr,
                    timestamp, filename_suffix, opts);
}

void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_35(mht_35_v, 1047, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloSnapshotIfEnabled");

  CanonicalDebugOptions opts(module.config().debug_options());
  if (!opts.should_dump_module(module.name()) || !opts.dump_snapshots) {
    return;
  }
  int64_t execution_count;
  uint64_t timestamp;
  {
    static auto& module_id_to_execution_count ABSL_GUARDED_BY(mu) =
        *new absl::flat_hash_map<int64_t, int64_t>();
    absl::MutexLock lock(&mu);
    execution_count = module_id_to_execution_count[module.unique_id()]++;
    auto timestamp_emplace = module_id_to_timestamp.try_emplace(
        module.unique_id(), tensorflow::Env::Default()->NowMicros());
    timestamp = timestamp_emplace.first->second;
  }
  std::string filename =
      StrCat(FilenameFor(module, std::to_string(timestamp),
                         StrFormat("execution_%04d", execution_count)),
             ".hlo_snapshot.pb");
  if (opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  std::string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, opts);
}

void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_36(mht_36_v, 1083, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloSnapshotIfEnabled");

  CanonicalDebugOptions canonical_opts(opts);
  std::string name = snapshot.hlo().hlo_module().name();
  if (!canonical_opts.should_dump_module(name) ||
      !canonical_opts.dump_snapshots) {
    return;
  }

  // We don't have a unique id for an HloSnapshot, so in this overload we just
  // have to use its name.
  int64_t execution_count;
  {
    static auto& module_name_to_execution_count ABSL_GUARDED_BY(mu) =
        *new absl::flat_hash_map<std::string, int64_t>();
    absl::MutexLock lock(&mu);
    execution_count = module_name_to_execution_count[name]++;
  }
  std::string filename = StrFormat("module_%s.execution_%04d.hlo_snapshot.pb",
                                   name, execution_count);
  if (canonical_opts.dumping_to_stdout()) {
    LOG(ERROR) << "Refusing to write HLO snapshot proto for " << filename
               << " to stdout.  Pass --xla_dump_to=<path> to write to a file.";
    return;
  }
  std::string pb;
  if (!tensorflow::SerializeToStringDeterministic(snapshot, &pb)) {
    LOG(ERROR) << "Failed to serialize HLO snapshot proto " << filename;
  }
  DumpToFileInDirImpl(filename, pb, canonical_opts);
}

void DumpHloModuleMetadataIfEnabled(const std::vector<HloModule*>& modules) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTcc mht_37(mht_37_v, 1117, "", "./tensorflow/compiler/xla/service/dump.cc", "DumpHloModuleMetadataIfEnabled");

  absl::flat_hash_set<int64_t> dumped_module_ids;
  for (const HloModule* module : modules) {
    CanonicalDebugOptions opts(module->config().debug_options());
    if (!opts.dump_module_metadata) {
      continue;
    }
    DumpHloModuleMetadata(module->metadata().proto(), opts, &dumped_module_ids);
    const absl::optional<HloModuleMetadataProto>& prepartitioning_metadata =
        module->metadata().prepartitioning_metadata();
    if (prepartitioning_metadata.has_value()) {
      DumpHloModuleMetadata(*prepartitioning_metadata, opts,
                            &dumped_module_ids);
    }
  }
}

}  // namespace xla
