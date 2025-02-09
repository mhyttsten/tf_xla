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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc() {
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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_code.h"

#include <stdio.h>

#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"

namespace tensorflow {
namespace tfprof {
namespace {

const char* const kGradientSuffix = " (gradient)";

// Convert to Trace proto into a short readable string.
std::string GetTraceString(const CallStack::Trace& trace) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "GetTraceString");

  std::string ntrace =
      absl::StrCat(io::Basename(trace.file()), ":", trace.lineno());
  if (trace.function().length() < 20) {
    absl::StrAppend(&ntrace, ":", trace.function());
  } else {
    absl::StrAppend(&ntrace, ":", trace.function().substr(0, 17), "...");
  }
  return ntrace;
}

bool IsGradNode(const string& name, string* forward_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "IsGradNode");

  // Given a forward operation with name op, its gradient op has the following
  // name: ...gradients/op_grad/...
  // TODO(xpan): This is hacky.
  auto grad_prefix = name.find("gradients/");
  auto grad_suffix = name.find("_grad/");
  if (grad_prefix == name.npos || grad_suffix == name.npos) {
    return false;
  }
  auto start = grad_prefix + string("gradients/").length();
  auto len = grad_suffix - start;
  if (len <= 0) {
    return false;
  }
  *forward_name = name.substr(start, len);
  return true;
}

// StringTable maps each string to an id.
class StringTable {
 public:
  StringTable() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "StringTable");

    // Pprof requires first entry in string_table to be ''.
    string_id_[""] = 0;
    all_strings_.push_back("");
  }

  // Returns the index of a string. If not found, inserts the string and
  // return the inserted index.
  uint64 GetIndex(const string& str) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("str: \"" + str + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "GetIndex");

    auto idx = string_id_.find(str);
    if (idx != string_id_.end()) {
      return idx->second;
    }
    all_strings_.push_back(str);
    return string_id_.insert(std::pair<string, int64_t>(str, string_id_.size()))
        .first->second;
  }

  const std::vector<string>& strings() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "strings");
 return all_strings_; }

 private:
  std::map<string, uint64> string_id_;
  std::vector<string> all_strings_;
};

// FunctionTable maps each function to an id.
class FunctionTable {
 public:
  explicit FunctionTable(StringTable* string_table)
      : string_table_(string_table) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "FunctionTable");
}

  // Returns the index of a function. If not found, adds a function proto
  // and returns the function index.
  uint64 GetIndex(const string& file_path, const string& func_name,
                  uint64 func_start_line) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file_path: \"" + file_path + "\"");
   mht_6_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "GetIndex");

    auto key = std::tuple<string, string, uint64>(file_path, func_name,
                                                  func_start_line);
    auto idx = function_table_.find(key);
    if (idx != function_table_.end()) {
      return idx->second.id();
    }
    pprof::Function* func_pb = &function_table_[key];
    // function index should start from 1.
    func_pb->set_id(function_table_.size());

    string file_base(io::Basename(file_path));
    file_base = file_base.substr(0, file_base.find_last_of('.'));
    func_pb->set_name(
        string_table_->GetIndex(absl::StrCat(file_base, ":", func_name)));
    func_pb->set_filename(string_table_->GetIndex(file_path));
    func_pb->set_start_line(func_start_line);
    return func_pb->id();
  }

  const std::map<std::tuple<string, string, uint64>, pprof::Function>&
  functions() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_7(mht_7_v, 320, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "functions");

    return function_table_;
  }

 private:
  StringTable* string_table_;
  std::map<std::tuple<string, string, uint64>, pprof::Function> function_table_;
};

// LocationTable maps each function call to an id.
class LocationTable {
 public:
  explicit LocationTable(FunctionTable* function_table)
      : function_table_(function_table) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_8(mht_8_v, 336, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "LocationTable");
}

  // Returns the index of a function call location. If not found, adds a
  // location proto and returns the location index.
  uint64 GetIndex(const string& file_path, uint64 line_number,
                  const string& called_function_name,
                  const string& called_file_path,
                  uint64 called_func_start_line) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("file_path: \"" + file_path + "\"");
   mht_9_v.push_back("called_function_name: \"" + called_function_name + "\"");
   mht_9_v.push_back("called_file_path: \"" + called_file_path + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_9(mht_9_v, 349, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "GetIndex");

    auto key = std::tuple<string, string, uint64>(
        file_path, called_function_name, line_number);

    auto idx = location_table_.find(key);
    if (idx != location_table_.end()) {
      return idx->second.id();
    }
    pprof::Location* location_pb = &location_table_[key];
    location_pb->set_id(location_table_.size());
    pprof::Line* line_pb = location_pb->add_line();
    line_pb->set_function_id(function_table_->GetIndex(
        called_file_path, called_function_name, called_func_start_line));
    line_pb->set_line(line_number);
    return location_pb->id();
  }

  const std::map<std::tuple<string, string, uint64>, pprof::Location>&
  locations() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_10(mht_10_v, 370, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "locations");

    return location_table_;
  }

 private:
  FunctionTable* function_table_;
  std::map<std::tuple<string, string, uint64>, pprof::Location> location_table_;
};

// Samples stores samples of all calls. A sample is a single call trace,
// that is, the call path from top caller to the leaf callee.
class Samples {
 public:
  explicit Samples(StringTable* string_table, const Options* opts)
      : string_table_(string_table), opts_(opts) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_11(mht_11_v, 387, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "Samples");
}

  // 'node' is the leaf of the displayed trace. It includes all graph nodes
  // created by it. 'location_ids' contains
  // the call stack, from callee to caller.
  // This method adds the statistics of graph nodes created by the python
  // call.
  void Add(const CodeNode* node, const std::vector<uint64>& location_ids) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_12(mht_12_v, 397, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "Add");

    // displayed leaf might not be true leaf. Retrieve the true leaves for
    // stats.
    std::vector<const CodeNode*> all_leaf = FetchAllLeaf(node);
    CHECK(!all_leaf.empty()) << node->name();

    for (const CodeNode* cn : all_leaf) {
      for (const auto& gn_it : cn->node->graph_nodes()) {
        const TFGraphNode* gn = gn_it.second;
        string name = gn->name();
        // Generate a new trace name, in case the name is taken.
        while (sample_table_.find(name) != sample_table_.end()) {
          name += '@';
        }
        pprof::Sample* sample_pb = &sample_table_[name];
        for (uint64 id : location_ids) {
          sample_pb->mutable_location_id()->Add(id);
        }
        pprof::Label* label_pb = sample_pb->mutable_label()->Add();
        label_pb->set_key(string_table_->GetIndex("graph node:"));
        label_pb->set_str(string_table_->GetIndex(gn->name()));

        sample_pb->mutable_value()->Add(1);
        string type = *opts_->select.begin();
        if (type == kShown[1]) {
          sample_pb->mutable_value()->Add(gn->exec_micros(node->node->step()));
        } else if (type == kShown[9]) {
          sample_pb->mutable_value()->Add(
              gn->accelerator_exec_micros(node->node->step()));
        } else if (type == kShown[10]) {
          sample_pb->mutable_value()->Add(
              gn->cpu_exec_micros(node->node->step()));
        } else if (type == kShown[0]) {
          sample_pb->mutable_value()->Add(
              gn->requested_bytes(node->node->step()));
        } else if (type == kShown[11]) {
          sample_pb->mutable_value()->Add(gn->peak_bytes(node->node->step()));
        } else if (type == kShown[12]) {
          sample_pb->mutable_value()->Add(
              gn->residual_bytes(node->node->step()));
        } else if (type == kShown[13]) {
          sample_pb->mutable_value()->Add(gn->output_bytes(node->node->step()));
        } else if (type == kShown[2]) {
          sample_pb->mutable_value()->Add(gn->parameters());
        } else if (type == kShown[3]) {
          sample_pb->mutable_value()->Add(gn->float_ops(node->node->step()));
        } else {
          absl::FPrintF(stderr, "pprof doesn't support -select=%s\n", type);
        }
      }
    }
  }

  const std::map<string, pprof::Sample>& samples() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_13(mht_13_v, 453, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "samples");

    return sample_table_;
  }

 private:
  std::vector<const CodeNode*> FetchAllLeaf(const CodeNode* root) {
    if (root->children.empty()) {
      return {root};
    }
    std::vector<const CodeNode*> ret;
    for (auto& n : root->children) {
      std::vector<const CodeNode*> nodes = FetchAllLeaf(n);
      ret.insert(ret.end(), nodes.begin(), nodes.end());
    }
    return ret;
  }

  StringTable* string_table_;
  const Options* opts_;
  std::map<string, pprof::Sample> sample_table_;
};

class PprofProfileImpl : public PprofProfile {
 public:
  explicit PprofProfileImpl(const Options* opts)
      : opts_(opts),
        func_table_(new FunctionTable(&string_table_)),
        loc_table_(new LocationTable(func_table_.get())),
        samples_(new Samples(&string_table_, opts)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_14(mht_14_v, 484, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "PprofProfileImpl");
}

  uint64 AddLocation(const CodeNode* callee, const CodeNode* caller) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_15(mht_15_v, 489, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "AddLocation");

    const string& file_path = caller->file();
    uint64 lineno = caller->lineno();
    const string& callee_file_path = callee->file();
    const string& callee_function = callee->function();
    uint64 callee_func_start_line = callee->func_start_line();

    return loc_table_->GetIndex(file_path, lineno, callee_function,
                                callee_file_path, callee_func_start_line);
  }

  void AddSample(const CodeNode* leaf, std::vector<uint64>* call_ids) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_16(mht_16_v, 503, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "AddSample");

    std::vector<uint64> reversed_call_ids;
    std::reverse_copy(call_ids->begin(), call_ids->end(),
                      std::back_inserter(reversed_call_ids));
    samples_->Add(leaf, reversed_call_ids);
  }

  Status WritePprofProfile(const string& filename) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_17(mht_17_v, 514, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "WritePprofProfile");

    pprof::Profile profile_pb;
    Build(&profile_pb);

    std::unique_ptr<WritableFile> file;
    Status s = Env::Default()->NewWritableFile(filename, &file);
    if (!s.ok()) return s;

    int32_t buf_size = 1024 * 1024;
    io::ZlibOutputBuffer* zlib_output_buffer = new io::ZlibOutputBuffer(
        file.get(), buf_size, buf_size, io::ZlibCompressionOptions::GZIP());
    s = zlib_output_buffer->Init();
    if (!s.ok()) {
      delete zlib_output_buffer;
      return s;
    }
    s = zlib_output_buffer->Append(profile_pb.SerializeAsString());
    if (!s.ok()) {
      delete zlib_output_buffer;
      return s;
    }
    s = zlib_output_buffer->Close();
    if (!s.ok()) {
      delete zlib_output_buffer;
      return s;
    }
    absl::FPrintF(stdout,
                  "\nRun pprof -png --nodecount=100 --sample_index=1 <%s>\n",
                  filename);
    delete zlib_output_buffer;
    return s;
  }

 private:
  void Build(pprof::Profile* profile_pb) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_18(mht_18_v, 551, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "Build");

    string sample_type_description = "count";
    auto sample_type = profile_pb->mutable_sample_type()->Add();
    sample_type->set_type(string_table_.GetIndex(sample_type_description));
    sample_type->set_unit(string_table_.GetIndex("count"));

    string type = *opts_->select.begin();
    sample_type_description = type;
    sample_type = profile_pb->mutable_sample_type()->Add();
    sample_type->set_type(string_table_.GetIndex(sample_type_description));
    if (type == kShown[1] || type == kShown[9] || type == kShown[10]) {
      sample_type->set_unit(string_table_.GetIndex("microseconds"));
      if (type == kShown[1]) {
        profile_pb->mutable_comment()->Add(string_table_.GetIndex(
            "Sum of accelerator execution time and cpu execution time."));
      } else if (type == kShown[9]) {
        profile_pb->mutable_comment()->Add(
            string_table_.GetIndex("Accelerator execution time."));
      } else if (type == kShown[10]) {
        profile_pb->mutable_comment()->Add(
            string_table_.GetIndex("CPU execution time."));
      }
    } else if (type == kShown[0]) {
      sample_type->set_unit(string_table_.GetIndex("bytes"));
      profile_pb->mutable_comment()->Add(
          string_table_.GetIndex("Sum of operation total memory requests, "
                                 "excluding deallocations."));
    } else if (type == kShown[11]) {
      sample_type->set_unit(string_table_.GetIndex("bytes"));
      profile_pb->mutable_comment()->Add(
          string_table_.GetIndex("Sum of operation peak memory usage."));
    } else if (type == kShown[12]) {
      sample_type->set_unit(string_table_.GetIndex("bytes"));
      profile_pb->mutable_comment()->Add(string_table_.GetIndex(
          "Sum of operation allocated memory after finish."));
    } else if (type == kShown[13]) {
      sample_type->set_unit(string_table_.GetIndex("bytes"));
      profile_pb->mutable_comment()->Add(
          string_table_.GetIndex("Sum of operation output size."));
    } else if (type == kShown[2]) {
      sample_type->set_unit(string_table_.GetIndex("count"));
      profile_pb->mutable_comment()->Add(
          string_table_.GetIndex("Model parameters."));
    } else if (type == kShown[3]) {
      sample_type->set_unit(string_table_.GetIndex("count"));
      profile_pb->mutable_comment()->Add(string_table_.GetIndex(
          "Model float operations (Only available if defined)."));
    } else {
      absl::FPrintF(stderr, "pprof doesn't support selecting: %s\n", type);
    }

    for (const string& str : string_table_.strings()) {
      *profile_pb->mutable_string_table()->Add() = str;
    }
    for (const auto& sample_it : samples_->samples()) {
      // TODO(xpan): Consider swap.
      profile_pb->mutable_sample()->Add()->MergeFrom(sample_it.second);
    }
    for (const auto& function_it : func_table_->functions()) {
      profile_pb->mutable_function()->Add()->MergeFrom(function_it.second);
    }
    for (const auto& location_it : loc_table_->locations()) {
      profile_pb->mutable_location()->Add()->MergeFrom(location_it.second);
    }
  }

  const Options* opts_;
  StringTable string_table_;
  std::unique_ptr<FunctionTable> func_table_;
  std::unique_ptr<LocationTable> loc_table_;
  std::unique_ptr<Samples> samples_;
};
}  // namespace

void TFCode::AddNode(TFGraphNode* node) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_19(mht_19_v, 628, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::AddNode");

  if (!node->call_stack() || node->call_stack()->traces().empty()) {
    return;
  }
  // We infer the forward operation name from gradient op name. So, we can
  // map gradient op traces to forward op traces.
  // E.g. gradient node of 'inp_1/Conv2D' would be 'gradients/inp_1/Conv2D_grad.
  string forward_name;
  if (IsGradNode(node->name(), &forward_name)) {
    auto grad_nodes_it = grad_nodes_.find(forward_name);
    if (grad_nodes_it != grad_nodes_.end()) {
      grad_nodes_it->second.push_back(node);
    } else {
      grad_nodes_.insert(
          std::pair<string, std::vector<TFGraphNode*>>(forward_name, {node}));
    }
    return;
  } else {
    forward_nodes_[node->name()] = node;
  }

  if (!root_) {
    graph_root_.reset(new TFMultiGraphNode(kTFProfRoot));
    root_.reset(new CodeNode(graph_root_.get(), nullptr, ""));
  }

  CodeNode* pre_code_node = root_.get();
  // TODO(xpan): Consider to release CodeDef after TFCode is built. It
  // takes a lot of memory.
  std::set<string> traces;
  for (int i = 0, end = node->call_stack()->traces().size(); i < end; ++i) {
    // Unlike op name, which is globally unique, trace name is only unique
    // w.r.t. it's parent.
    const string& trace = GetTraceString(node->call_stack()->traces().at(i));
    traces.insert(trace);
    pre_code_node = pre_code_node->AddChildren(
        trace, &node->call_stack()->traces().at(i), "");
    const int64_t last_index = node->call_stack()->traces().size() - 1;
    if (i == last_index) {
      pre_code_node->node->AddGraphNode(node);
    }
  }
}

void TFCode::Build() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_20(mht_20_v, 675, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::Build");

  int64_t unaccounted_nodes = 0;
  for (const auto& it : grad_nodes_) {
    const string& forward_name = it.first;
    auto forward_it = forward_nodes_.find(forward_name);
    if (forward_it == forward_nodes_.end()) {
      unaccounted_nodes += 1;
      continue;
    }
    TFGraphNode* fn = forward_it->second;
    CodeNode* leaf = nullptr;
    CodeNode* pre_code_node = root_.get();
    for (int i = 0, end = fn->call_stack()->traces().size(); i < end; ++i) {
      const string& trace =
          GetTraceString(fn->call_stack()->traces().at(i)) + kGradientSuffix;
      pre_code_node = pre_code_node->AddChildren(
          trace, &fn->call_stack()->traces().at(i), kGradientSuffix);
      const int64_t last_trace = fn->call_stack()->traces().size() - 1;
      if (i == last_trace) {
        leaf = pre_code_node;
      }
    }
    for (TFGraphNode* gn : it.second) {
      leaf->node->AddGraphNode(gn);
    }
  }
  if (unaccounted_nodes > 0) {
    absl::FPrintF(stderr, "%d gradient nodes not accounted\n",
                  unaccounted_nodes);
  }
}

const ShowMultiNode* TFCode::ShowInternal(const Options& opts,
                                          Timeline* timeline) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_21(mht_21_v, 711, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::ShowInternal");

  root_->ResetTotalStats();
  if (opts.output_type == kOutput[3]) {
    if (opts.select.size() != 1) {
      absl::FPrintF(stderr, "Can only select 1 attribute for pprof output.\n");
      return root_.get();
    }
    string select = *opts.select.begin();
    if (select != kShown[0] && select != kShown[1] && select != kShown[2] &&
        select != kShown[3] && select != kShown[9] && select != kShown[10] &&
        select != kShown[11] && select != kShown[12] && select != kShown[13]) {
      absl::FPrintF(stderr, "pprof doesn't support -select=%s\n", select);
      return root_.get();
    }
  }
  if (opts.account_displayed_op_only) {
    absl::FPrintF(stderr,
                  "Note: code view ignores account_displayed_op_only\n");
  }

  std::vector<CodeNode*> roots = Account(root_->children, opts);
  root_->show_children.clear();
  for (CodeNode* n : roots) {
    root_->AggregateTotalStats(n);
  }

  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    roots = SearchRoot(roots, opts.start_name_regexes);
  }

  root_->show_children.assign(roots.begin(), roots.end());

  CodeNode* root = PrintScope({root_.get()}, opts, 1, 0)[0];

  root->formatted_str = FormatLegend(opts) + root->formatted_str;

  if (opts.output_type == kOutput[3]) {
    std::vector<uint64> call_ids;
    pprof_profile_.reset(new PprofProfileImpl(&opts));
    Format(root, root->show_children, opts, &root->formatted_str,
           root->mutable_proto(), &call_ids);
    Status s = pprof_profile_->WritePprofProfile(
        opts.output_options.at(kPprofOpts[0]));
    if (!s.ok()) {
      absl::FPrintF(stderr, "%s\n", s.ToString());
    }
  } else {
    Format(root, root->show_children, opts, &root->formatted_str,
           root->mutable_proto(), nullptr);
    if (timeline) {
      timeline->GenerateCodeTimeline(root);
    }
  }
  return root;
}

void TFCode::Format(const CodeNode* root, const std::vector<CodeNode*>& nodes,
                    const Options& opts, string* display_str,
                    MultiGraphNodeProto* proto, std::vector<uint64>* call_ids) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_22(mht_22_v, 773, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::Format");

  if (nodes.empty() && root->has_trace() && opts.output_type == kOutput[3]) {
    pprof_profile_->AddSample(root, call_ids);
  }

  for (CodeNode* node : nodes) {
    if (root->has_trace() && opts.output_type == kOutput[3]) {
      uint64 loc_id = pprof_profile_->AddLocation(node, root);
      call_ids->push_back(loc_id);
    }
    display_str->append(node->formatted_str);
    MultiGraphNodeProto* child = proto->add_children();
    child->MergeFrom(node->proto());
    Format(node, node->show_children, opts, display_str, child, call_ids);
    if (root->has_trace() && opts.output_type == kOutput[3]) {
      call_ids->pop_back();
    }
  }
}

std::vector<CodeNode*> TFCode::SearchRoot(std::vector<CodeNode*> roots,
                                          const std::vector<string>& regexes) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_23(mht_23_v, 797, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::SearchRoot");

  std::vector<CodeNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (CodeNode* root : roots) {
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      // Found a start node at this branch, no need to continue.
      continue;
    }
    std::vector<CodeNode*> nroots = SearchRoot(root->show_children, regexes);
    res.insert(res.end(), nroots.begin(), nroots.end());
  }
  return res;
}

std::vector<CodeNode*> TFCode::PrintScope(const std::vector<CodeNode*> roots,
                                          const Options& opts, int depth,
                                          int last_ident) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_24(mht_24_v, 826, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::PrintScope");

  std::vector<CodeNode*> show_nodes;

  for (CodeNode* node : roots) {
    if (ShouldTrim(node, opts.trim_name_regexes) || depth > opts.max_depth) {
      continue;
    }
    int ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) ident += 2;

    std::vector<CodeNode*> show_cnodes =
        PrintScope(node->show_children, opts, depth + 1, ident);
    if (show) {
      node->show_children.clear();

      show_cnodes = SortNodes(show_cnodes, opts);
      for (CodeNode* sc : show_cnodes) {
        node->show_children.push_back(sc);
      }

      node->formatted_str = FormatNode(node, opts, last_ident);

      if (opts.select.find(kShown[4]) != opts.select.end()) {
        absl::FPrintF(stderr, "code view has no tensor value to show\n");
      }
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

std::vector<CodeNode*> TFCode::Account(const std::vector<CodeNode*>& roots,
                                       const Options& opts) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_25(mht_25_v, 865, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::Account");

  std::vector<CodeNode*> act_nodes;

  for (CodeNode* node : roots) {
    node->ResetTotalStats();
    std::vector<CodeNode*> act_cnodes = Account(node->children, opts);
    node->account = ReAccount(node, opts);
    if (node->account || !act_cnodes.empty()) {
      node->show_children.clear();
      node->ResetTotalStats();
      node->AddSelfToTotalStats();
      for (CodeNode* c : act_cnodes) {
        node->AggregateTotalStats(c);
        node->show_children.push_back(c);
      }
      act_nodes.push_back(node);
    }
  }
  return act_nodes;
}

string TFCode::FormatNodeMemory(CodeNode* node, int64_t bytes,
                                int64_t total_bytes) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_26(mht_26_v, 890, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::FormatNodeMemory");

  string memory = FormatMemory(total_bytes);
  if (node->account) {
    memory = FormatMemory(bytes) + "/" + memory;
  } else {
    memory = "--/" + memory;
  }
  return memory;
}

string TFCode::FormatNode(CodeNode* node, const Options& opts,
                          int64_t indent) const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_codeDTcc mht_27(mht_27_v, 904, "", "./tensorflow/core/profiler/internal/tfprof_code.cc", "TFCode::FormatNode");

  std::vector<string> attrs;
  if (opts.select.find(kShown[0]) != opts.select.end()) {
    attrs.push_back(FormatNodeMemory(node, node->proto().requested_bytes(),
                                     node->proto().total_requested_bytes()));
  }
  if (opts.select.find(kShown[11]) != opts.select.end()) {
    attrs.push_back(FormatNodeMemory(node, node->proto().peak_bytes(),
                                     node->proto().total_peak_bytes()));
  }
  if (opts.select.find(kShown[12]) != opts.select.end()) {
    attrs.push_back(FormatNodeMemory(node, node->proto().residual_bytes(),
                                     node->proto().total_residual_bytes()));
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    attrs.push_back(FormatNodeMemory(node, node->proto().output_bytes(),
                                     node->proto().total_output_bytes()));
  }

  std::vector<string> time_attrs = FormatTimes(node, opts);
  attrs.insert(attrs.end(), time_attrs.begin(), time_attrs.end());

  if (opts.select.find(kShown[2]) != opts.select.end()) {
    string params = FormatNumber(node->proto().total_parameters()) + " params";
    if (node->account) {
      params = FormatNumber(node->proto().parameters()) + "/" + params;
    } else {
      params = "--/" + params;
    }
    attrs.push_back(params);
  }

  if (opts.select.find(kShown[3]) != opts.select.end()) {
    string fops = FormatNumber(node->proto().total_float_ops()) + " flops";
    if (node->account) {
      fops = FormatNumber(node->proto().float_ops()) + "/" + fops;
    } else {
      fops = "--/" + fops;
    }
    attrs.push_back(fops);
  }

  if (opts.select.find(kShown[5]) != opts.select.end() &&
      !node->node->devices().empty()) {
    attrs.push_back(absl::StrJoin(node->node->devices(), "|"));
  }
  if (opts.select.find(kShown[6]) != opts.select.end()) {
    std::set<string> op_types = node->node->op_types();
    attrs.push_back(absl::StrJoin(op_types, "|"));
  }
  if (opts.select.find(kShown[7]) != opts.select.end()) {
    // TODO(xpan): Make op count available in code view?
    attrs.push_back(absl::StrFormat("%s N/A in code view", kShown[7]));
  }
  if (opts.select.find(kShown[8]) != opts.select.end()) {
    attrs.push_back(absl::StrFormat("%s N/A in code view", kShown[8]));
  }

  return absl::StrFormat("%s%s (%s)\n", std::string(indent, ' '), node->name(),
                         absl::StrJoin(attrs, ", "));
}
}  // namespace tfprof
}  // namespace tensorflow
