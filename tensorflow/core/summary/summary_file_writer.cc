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
class MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/summary/summary_file_writer.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/summary/summary_converter.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

class SummaryFileWriter : public SummaryWriterInterface {
 public:
  SummaryFileWriter(int max_queue, int flush_millis, Env* env)
      : SummaryWriterInterface(),
        is_initialized_(false),
        max_queue_(max_queue),
        flush_millis_(flush_millis),
        env_(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/summary/summary_file_writer.cc", "SummaryFileWriter");
}

  Status Initialize(const string& logdir, const string& filename_suffix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("logdir: \"" + logdir + "\"");
   mht_1_v.push_back("filename_suffix: \"" + filename_suffix + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/summary/summary_file_writer.cc", "Initialize");

    const Status is_dir = env_->IsDirectory(logdir);
    if (!is_dir.ok()) {
      if (is_dir.code() != tensorflow::error::NOT_FOUND) {
        return is_dir;
      }
      TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(logdir));
    }
    // Embed PID plus a unique counter as the leading portion of the filename
    // suffix to help prevent filename collisions between and within processes.
    int32_t pid = env_->GetProcessId();
    static std::atomic<int64_t> file_id_counter(0);
    // Precede filename_suffix with "." if it doesn't already start with one.
    string sep = absl::StartsWith(filename_suffix, ".") ? "" : ".";
    const string uniquified_filename_suffix = absl::StrCat(
        ".", pid, ".", file_id_counter.fetch_add(1), sep, filename_suffix);
    mutex_lock ml(mu_);
    events_writer_ =
        tensorflow::MakeUnique<EventsWriter>(io::JoinPath(logdir, "events"));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        events_writer_->InitWithSuffix(uniquified_filename_suffix),
        "Could not initialize events writer.");
    last_flush_ = env_->NowMicros();
    is_initialized_ = true;
    return Status::OK();
  }

  Status Flush() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/summary/summary_file_writer.cc", "Flush");

    mutex_lock ml(mu_);
    if (!is_initialized_) {
      return errors::FailedPrecondition("Class was not properly initialized.");
    }
    return InternalFlush();
  }

  ~SummaryFileWriter() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/summary/summary_file_writer.cc", "~SummaryFileWriter");

    (void)Flush();  // Ignore errors.
  }

  Status WriteTensor(int64_t global_step, Tensor t, const string& tag,
                     const string& serialized_metadata) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tag: \"" + tag + "\"");
   mht_4_v.push_back("serialized_metadata: \"" + serialized_metadata + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteTensor");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    Summary::Value* v = e->mutable_summary()->add_value();

    if (t.dtype() == DT_STRING) {
      // Treat DT_STRING specially, so that tensor_util.MakeNdarray in Python
      // can convert the TensorProto to string-type numpy array. MakeNdarray
      // does not work with strings encoded by AsProtoTensorContent() in
      // tensor_content.
      t.AsProtoField(v->mutable_tensor());
    } else {
      t.AsProtoTensorContent(v->mutable_tensor());
    }
    v->set_tag(tag);
    if (!serialized_metadata.empty()) {
      v->mutable_metadata()->ParseFromString(serialized_metadata);
    }
    return WriteEvent(std::move(e));
  }

  Status WriteScalar(int64_t global_step, Tensor t,
                     const string& tag) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteScalar");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(
        AddTensorAsScalarToSummary(t, tag, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteHistogram(int64_t global_step, Tensor t,
                        const string& tag) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_6(mht_6_v, 307, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteHistogram");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(
        AddTensorAsHistogramToSummary(t, tag, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteImage(int64_t global_step, Tensor t, const string& tag,
                    int max_images, Tensor bad_color) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_7(mht_7_v, 321, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteImage");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(AddTensorAsImageToSummary(t, tag, max_images, bad_color,
                                                 e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteAudio(int64_t global_step, Tensor t, const string& tag,
                    int max_outputs, float sample_rate) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_8(mht_8_v, 335, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteAudio");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    TF_RETURN_IF_ERROR(AddTensorAsAudioToSummary(
        t, tag, max_outputs, sample_rate, e->mutable_summary()));
    return WriteEvent(std::move(e));
  }

  Status WriteGraph(int64_t global_step,
                    std::unique_ptr<GraphDef> graph) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_9(mht_9_v, 348, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteGraph");

    std::unique_ptr<Event> e{new Event};
    e->set_step(global_step);
    e->set_wall_time(GetWallTime());
    graph->SerializeToString(e->mutable_graph_def());
    return WriteEvent(std::move(e));
  }

  Status WriteEvent(std::unique_ptr<Event> event) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_10(mht_10_v, 359, "", "./tensorflow/core/summary/summary_file_writer.cc", "WriteEvent");

    mutex_lock ml(mu_);
    queue_.emplace_back(std::move(event));
    if (queue_.size() > max_queue_ ||
        env_->NowMicros() - last_flush_ > 1000 * flush_millis_) {
      return InternalFlush();
    }
    return Status::OK();
  }

  string DebugString() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_11(mht_11_v, 372, "", "./tensorflow/core/summary/summary_file_writer.cc", "DebugString");
 return "SummaryFileWriter"; }

 private:
  double GetWallTime() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_12(mht_12_v, 378, "", "./tensorflow/core/summary/summary_file_writer.cc", "GetWallTime");

    return static_cast<double>(env_->NowMicros()) / 1.0e6;
  }

  Status InternalFlush() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_13(mht_13_v, 385, "", "./tensorflow/core/summary/summary_file_writer.cc", "InternalFlush");

    for (const std::unique_ptr<Event>& e : queue_) {
      events_writer_->WriteEvent(*e);
    }
    queue_.clear();
    TF_RETURN_WITH_CONTEXT_IF_ERROR(events_writer_->Flush(),
                                    "Could not flush events file.");
    last_flush_ = env_->NowMicros();
    return Status::OK();
  }

  bool is_initialized_;
  const int max_queue_;
  const int flush_millis_;
  uint64 last_flush_;
  Env* env_;
  mutex mu_;
  std::vector<std::unique_ptr<Event>> queue_ TF_GUARDED_BY(mu_);
  // A pointer to allow deferred construction.
  std::unique_ptr<EventsWriter> events_writer_ TF_GUARDED_BY(mu_);
  std::vector<std::pair<string, SummaryMetadata>> registered_summaries_
      TF_GUARDED_BY(mu_);
};

}  // namespace

Status CreateSummaryFileWriter(int max_queue, int flush_millis,
                               const string& logdir,
                               const string& filename_suffix, Env* env,
                               SummaryWriterInterface** result) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("logdir: \"" + logdir + "\"");
   mht_14_v.push_back("filename_suffix: \"" + filename_suffix + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writerDTcc mht_14(mht_14_v, 419, "", "./tensorflow/core/summary/summary_file_writer.cc", "CreateSummaryFileWriter");

  SummaryFileWriter* w = new SummaryFileWriter(max_queue, flush_millis, env);
  const Status s = w->Initialize(logdir, filename_suffix);
  if (!s.ok()) {
    w->Unref();
    *result = nullptr;
    return s;
  }
  *result = w;
  return Status::OK();
}

}  // namespace tensorflow
