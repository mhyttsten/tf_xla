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
class MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc() {
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

#include "tensorflow/core/util/debug_events_writer.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tfdbg {

namespace {
void MaybeSetDebugEventTimestamp(DebugEvent* debug_event, Env* env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/util/debug_events_writer.cc", "MaybeSetDebugEventTimestamp");

  if (debug_event->wall_time() == 0) {
    debug_event->set_wall_time(env->NowMicros() / 1e6);
  }
}
}  // namespace

SingleDebugEventFileWriter::SingleDebugEventFileWriter(const string& file_path)
    : env_(Env::Default()),
      file_path_(file_path),
      num_outstanding_events_(0),
      writer_mu_() {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("file_path: \"" + file_path + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::SingleDebugEventFileWriter");
}

Status SingleDebugEventFileWriter::Init() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_2(mht_2_v, 217, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::Init");

  if (record_writer_ != nullptr) {
    // TODO(cais): We currently don't check for file deletion. When the need
    // arises, check and fix it.
    return Status::OK();
  }

  // Reset recordio_writer (which has a reference to writable_file_) so final
  // Flush() and Close() call have access to writable_file_.
  record_writer_.reset();

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      env_->NewWritableFile(file_path_, &writable_file_),
      "Creating writable file ", file_path_);
  record_writer_.reset(new io::RecordWriter(writable_file_.get()));
  if (record_writer_ == nullptr) {
    return errors::Unknown("Could not create record writer at path: ",
                           file_path_);
  }
  num_outstanding_events_.store(0);
  VLOG(1) << "Successfully opened debug events file: " << file_path_;
  return Status::OK();
}

void SingleDebugEventFileWriter::WriteSerializedDebugEvent(
    StringPiece debug_event_str) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::WriteSerializedDebugEvent");

  if (record_writer_ == nullptr) {
    if (!Init().ok()) {
      LOG(ERROR) << "Write failed because file could not be opened.";
      return;
    }
  }
  num_outstanding_events_.fetch_add(1);
  {
    mutex_lock l(writer_mu_);
    record_writer_->WriteRecord(debug_event_str).IgnoreError();
  }
}

Status SingleDebugEventFileWriter::Flush() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::Flush");

  const int num_outstanding = num_outstanding_events_.load();
  if (num_outstanding == 0) {
    return Status::OK();
  }
  if (writable_file_ == nullptr) {
    return errors::Unknown("Unexpected NULL file for path: ", file_path_);
  }

  {
    mutex_lock l(writer_mu_);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(record_writer_->Flush(), "Failed to flush ",
                                    num_outstanding, " debug events to ",
                                    file_path_);
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(writable_file_->Sync(), "Failed to sync ",
                                  num_outstanding, " debug events to ",
                                  file_path_);
  num_outstanding_events_.store(0);
  return Status::OK();
}

Status SingleDebugEventFileWriter::Close() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::Close");

  Status status = Flush();
  if (writable_file_ != nullptr) {
    Status close_status = writable_file_->Close();
    if (!close_status.ok()) {
      status = close_status;
    }
    record_writer_.reset(nullptr);
    writable_file_.reset(nullptr);
  }
  num_outstanding_events_ = 0;
  return status;
}

const string SingleDebugEventFileWriter::FileName() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/util/debug_events_writer.cc", "SingleDebugEventFileWriter::FileName");
 return file_path_; }

mutex DebugEventsWriter::factory_mu_(LINKER_INITIALIZED);

DebugEventsWriter::~DebugEventsWriter() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_7(mht_7_v, 312, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::~DebugEventsWriter");
 Close().IgnoreError(); }

// static
DebugEventsWriter* DebugEventsWriter::GetDebugEventsWriter(
    const string& dump_root, const string& tfdbg_run_id,
    int64_t circular_buffer_size) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("dump_root: \"" + dump_root + "\"");
   mht_8_v.push_back("tfdbg_run_id: \"" + tfdbg_run_id + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_8(mht_8_v, 322, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::GetDebugEventsWriter");

  mutex_lock l(DebugEventsWriter::factory_mu_);
  std::unordered_map<string, std::unique_ptr<DebugEventsWriter>>* writer_pool =
      DebugEventsWriter::GetDebugEventsWriterMap();
  if (writer_pool->find(dump_root) == writer_pool->end()) {
    std::unique_ptr<DebugEventsWriter> writer(
        new DebugEventsWriter(dump_root, tfdbg_run_id, circular_buffer_size));
    writer_pool->insert(std::make_pair(dump_root, std::move(writer)));
  }
  return (*writer_pool)[dump_root].get();
}

// static
Status DebugEventsWriter::LookUpDebugEventsWriter(
    const string& dump_root, DebugEventsWriter** debug_events_writer) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("dump_root: \"" + dump_root + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_9(mht_9_v, 340, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::LookUpDebugEventsWriter");

  mutex_lock l(DebugEventsWriter::factory_mu_);
  std::unordered_map<string, std::unique_ptr<DebugEventsWriter>>* writer_pool =
      DebugEventsWriter::GetDebugEventsWriterMap();
  if (writer_pool->find(dump_root) == writer_pool->end()) {
    return errors::FailedPrecondition(
        "No DebugEventsWriter has been created at dump root ", dump_root);
  }
  *debug_events_writer = (*writer_pool)[dump_root].get();
  return Status::OK();
}

Status DebugEventsWriter::Init() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_10(mht_10_v, 355, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::Init");

  mutex_lock l(initialization_mu_);

  // TODO(cais): We currently don't check for file deletion. When the need
  // arises, check and fix file deletion.
  if (is_initialized_) {
    return Status::OK();
  }

  if (!env_->IsDirectory(dump_root_).ok()) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(env_->RecursivelyCreateDir(dump_root_),
                                    "Failed to create directory ", dump_root_);
  }

  int64_t time_in_seconds = env_->NowMicros() / 1e6;
  file_prefix_ = io::JoinPath(
      dump_root_, strings::Printf("%s.%010lld.%s", kFileNamePrefix,
                                  static_cast<long long>(time_in_seconds),
                                  port::Hostname().c_str()));
  TF_RETURN_IF_ERROR(InitNonMetadataFile(SOURCE_FILES));
  TF_RETURN_IF_ERROR(InitNonMetadataFile(STACK_FRAMES));
  TF_RETURN_IF_ERROR(InitNonMetadataFile(GRAPHS));

  // In case there is one left over from before.
  metadata_writer_.reset();

  // The metadata file should be created.
  string metadata_filename = GetFileNameInternal(METADATA);
  metadata_writer_.reset(new SingleDebugEventFileWriter(metadata_filename));
  if (metadata_writer_ == nullptr) {
    return errors::Unknown("Could not create debug event metadata file writer");
  }

  DebugEvent debug_event;
  DebugMetadata* metadata = debug_event.mutable_debug_metadata();
  metadata->set_tensorflow_version(TF_VERSION_STRING);
  metadata->set_file_version(
      strings::Printf("%s%d", kVersionPrefix, kCurrentFormatVersion));
  metadata->set_tfdbg_run_id(tfdbg_run_id_);
  TF_RETURN_IF_ERROR(SerializeAndWriteDebugEvent(&debug_event, METADATA));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      metadata_writer_->Flush(), "Failed to flush debug event metadata writer");

  TF_RETURN_IF_ERROR(InitNonMetadataFile(EXECUTION));
  TF_RETURN_IF_ERROR(InitNonMetadataFile(GRAPH_EXECUTION_TRACES));
  is_initialized_ = true;
  return Status::OK();
}

Status DebugEventsWriter::WriteSourceFile(SourceFile* source_file) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_11(mht_11_v, 407, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteSourceFile");

  DebugEvent debug_event;
  debug_event.set_allocated_source_file(source_file);
  return SerializeAndWriteDebugEvent(&debug_event, SOURCE_FILES);
}

Status DebugEventsWriter::WriteStackFrameWithId(
    StackFrameWithId* stack_frame_with_id) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_12(mht_12_v, 417, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteStackFrameWithId");

  DebugEvent debug_event;
  debug_event.set_allocated_stack_frame_with_id(stack_frame_with_id);
  return SerializeAndWriteDebugEvent(&debug_event, STACK_FRAMES);
}

Status DebugEventsWriter::WriteGraphOpCreation(
    GraphOpCreation* graph_op_creation) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_13(mht_13_v, 427, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteGraphOpCreation");

  DebugEvent debug_event;
  debug_event.set_allocated_graph_op_creation(graph_op_creation);
  return SerializeAndWriteDebugEvent(&debug_event, GRAPHS);
}

Status DebugEventsWriter::WriteDebuggedGraph(DebuggedGraph* debugged_graph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_14(mht_14_v, 436, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteDebuggedGraph");

  DebugEvent debug_event;
  debug_event.set_allocated_debugged_graph(debugged_graph);
  return SerializeAndWriteDebugEvent(&debug_event, GRAPHS);
}

Status DebugEventsWriter::WriteExecution(Execution* execution) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_15(mht_15_v, 445, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteExecution");

  if (circular_buffer_size_ <= 0) {
    // No cyclic-buffer behavior.
    DebugEvent debug_event;
    debug_event.set_allocated_execution(execution);
    return SerializeAndWriteDebugEvent(&debug_event, EXECUTION);
  } else {
    // Circular buffer behavior.
    DebugEvent debug_event;
    MaybeSetDebugEventTimestamp(&debug_event, env_);
    debug_event.set_allocated_execution(execution);
    string serialized;
    debug_event.SerializeToString(&serialized);

    mutex_lock l(execution_buffer_mu_);
    execution_buffer_.emplace_back(std::move(serialized));
    if (execution_buffer_.size() > circular_buffer_size_) {
      execution_buffer_.pop_front();
    }
    return Status::OK();
  }
}

Status DebugEventsWriter::WriteGraphExecutionTrace(
    GraphExecutionTrace* graph_execution_trace) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_16(mht_16_v, 472, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteGraphExecutionTrace");

  TF_RETURN_IF_ERROR(Init());
  if (circular_buffer_size_ <= 0) {
    // No cyclic-buffer behavior.
    DebugEvent debug_event;
    debug_event.set_allocated_graph_execution_trace(graph_execution_trace);
    return SerializeAndWriteDebugEvent(&debug_event, GRAPH_EXECUTION_TRACES);
  } else {
    // Circular buffer behavior.
    DebugEvent debug_event;
    MaybeSetDebugEventTimestamp(&debug_event, env_);
    debug_event.set_allocated_graph_execution_trace(graph_execution_trace);
    string serialized;
    debug_event.SerializeToString(&serialized);

    mutex_lock l(graph_execution_trace_buffer_mu_);
    graph_execution_trace_buffer_.emplace_back(std::move(serialized));
    if (graph_execution_trace_buffer_.size() > circular_buffer_size_) {
      graph_execution_trace_buffer_.pop_front();
    }
    return Status::OK();
  }
}

Status DebugEventsWriter::WriteGraphExecutionTrace(
    const string& tfdbg_context_id, const string& device_name,
    const string& op_name, int32_t output_slot, int32_t tensor_debug_mode,
    const Tensor& tensor_value) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("tfdbg_context_id: \"" + tfdbg_context_id + "\"");
   mht_17_v.push_back("device_name: \"" + device_name + "\"");
   mht_17_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_17(mht_17_v, 505, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteGraphExecutionTrace");

  std::unique_ptr<GraphExecutionTrace> trace(new GraphExecutionTrace());
  trace->set_tfdbg_context_id(tfdbg_context_id);
  if (!op_name.empty()) {
    trace->set_op_name(op_name);
  }
  if (output_slot > 0) {
    trace->set_output_slot(output_slot);
  }
  if (tensor_debug_mode > 0) {
    trace->set_tensor_debug_mode(TensorDebugMode(tensor_debug_mode));
  }
  trace->set_device_name(device_name);
  tensor_value.AsProtoTensorContent(trace->mutable_tensor_proto());
  return WriteGraphExecutionTrace(trace.release());
}

void DebugEventsWriter::WriteSerializedNonExecutionDebugEvent(
    const string& debug_event_str, DebugEventFileType type) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("debug_event_str: \"" + debug_event_str + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_18(mht_18_v, 527, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteSerializedNonExecutionDebugEvent");

  std::unique_ptr<SingleDebugEventFileWriter>* writer = nullptr;
  SelectWriter(type, &writer);
  (*writer)->WriteSerializedDebugEvent(debug_event_str);
}

void DebugEventsWriter::WriteSerializedExecutionDebugEvent(
    const string& debug_event_str, DebugEventFileType type) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("debug_event_str: \"" + debug_event_str + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_19(mht_19_v, 538, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::WriteSerializedExecutionDebugEvent");

  const std::unique_ptr<SingleDebugEventFileWriter>* writer = nullptr;
  std::deque<string>* buffer = nullptr;
  mutex* mu = nullptr;
  switch (type) {
    case EXECUTION:
      writer = &execution_writer_;
      buffer = &execution_buffer_;
      mu = &execution_buffer_mu_;
      break;
    case GRAPH_EXECUTION_TRACES:
      writer = &graph_execution_traces_writer_;
      buffer = &graph_execution_trace_buffer_;
      mu = &graph_execution_trace_buffer_mu_;
      break;
    default:
      return;
  }

  if (circular_buffer_size_ <= 0) {
    // No cyclic-buffer behavior.
    (*writer)->WriteSerializedDebugEvent(debug_event_str);
  } else {
    // Circular buffer behavior.
    mutex_lock l(*mu);
    buffer->push_back(debug_event_str);
    if (buffer->size() > circular_buffer_size_) {
      buffer->pop_front();
    }
  }
}

int DebugEventsWriter::RegisterDeviceAndGetId(const string& device_name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_20(mht_20_v, 574, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::RegisterDeviceAndGetId");

  mutex_lock l(device_mu_);
  int& device_id = device_name_to_id_[device_name];
  if (device_id == 0) {
    device_id = device_name_to_id_.size();
    DebugEvent debug_event;
    MaybeSetDebugEventTimestamp(&debug_event, env_);
    DebuggedDevice* debugged_device = debug_event.mutable_debugged_device();
    debugged_device->set_device_name(device_name);
    debugged_device->set_device_id(device_id);
    string serialized;
    debug_event.SerializeToString(&serialized);
    graphs_writer_->WriteSerializedDebugEvent(serialized);
  }
  return device_id;
}

Status DebugEventsWriter::FlushNonExecutionFiles() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_21(mht_21_v, 594, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::FlushNonExecutionFiles");

  TF_RETURN_IF_ERROR(Init());
  if (source_files_writer_ != nullptr) {
    TF_RETURN_IF_ERROR(source_files_writer_->Flush());
  }
  if (stack_frames_writer_ != nullptr) {
    TF_RETURN_IF_ERROR(stack_frames_writer_->Flush());
  }
  if (graphs_writer_ != nullptr) {
    TF_RETURN_IF_ERROR(graphs_writer_->Flush());
  }
  return Status::OK();
}

Status DebugEventsWriter::FlushExecutionFiles() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_22(mht_22_v, 611, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::FlushExecutionFiles");

  TF_RETURN_IF_ERROR(Init());

  if (execution_writer_ != nullptr) {
    if (circular_buffer_size_ > 0) {
      // Write out all the content in the circular buffers.
      mutex_lock l(execution_buffer_mu_);
      while (!execution_buffer_.empty()) {
        execution_writer_->WriteSerializedDebugEvent(execution_buffer_.front());
        // SerializeAndWriteDebugEvent(&execution_buffer_.front());
        execution_buffer_.pop_front();
      }
    }
    TF_RETURN_IF_ERROR(execution_writer_->Flush());
  }

  if (graph_execution_traces_writer_ != nullptr) {
    if (circular_buffer_size_ > 0) {
      // Write out all the content in the circular buffers.
      mutex_lock l(graph_execution_trace_buffer_mu_);
      while (!graph_execution_trace_buffer_.empty()) {
        graph_execution_traces_writer_->WriteSerializedDebugEvent(
            graph_execution_trace_buffer_.front());
        graph_execution_trace_buffer_.pop_front();
      }
    }
    TF_RETURN_IF_ERROR(graph_execution_traces_writer_->Flush());
  }

  return Status::OK();
}

string DebugEventsWriter::FileName(DebugEventFileType type) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_23(mht_23_v, 646, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::FileName");

  if (file_prefix_.empty()) {
    Init().IgnoreError();
  }
  return GetFileNameInternal(type);
}

Status DebugEventsWriter::Close() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_24(mht_24_v, 656, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::Close");

  {
    mutex_lock l(initialization_mu_);
    if (!is_initialized_) {
      return Status::OK();
    }
  }

  std::vector<string> failed_to_close_files;

  if (metadata_writer_ != nullptr) {
    if (!metadata_writer_->Close().ok()) {
      failed_to_close_files.push_back(metadata_writer_->FileName());
    }
    metadata_writer_.reset(nullptr);
  }

  TF_RETURN_IF_ERROR(FlushNonExecutionFiles());
  if (source_files_writer_ != nullptr) {
    if (!source_files_writer_->Close().ok()) {
      failed_to_close_files.push_back(source_files_writer_->FileName());
    }
    source_files_writer_.reset(nullptr);
  }
  if (stack_frames_writer_ != nullptr) {
    if (!stack_frames_writer_->Close().ok()) {
      failed_to_close_files.push_back(stack_frames_writer_->FileName());
    }
    stack_frames_writer_.reset(nullptr);
  }
  if (graphs_writer_ != nullptr) {
    if (!graphs_writer_->Close().ok()) {
      failed_to_close_files.push_back(graphs_writer_->FileName());
    }
    graphs_writer_.reset(nullptr);
  }

  TF_RETURN_IF_ERROR(FlushExecutionFiles());
  if (execution_writer_ != nullptr) {
    if (!execution_writer_->Close().ok()) {
      failed_to_close_files.push_back(execution_writer_->FileName());
    }
    execution_writer_.reset(nullptr);
  }
  if (graph_execution_traces_writer_ != nullptr) {
    if (!graph_execution_traces_writer_->Close().ok()) {
      failed_to_close_files.push_back(
          graph_execution_traces_writer_->FileName());
    }
    graph_execution_traces_writer_.reset(nullptr);
  }

  if (failed_to_close_files.empty()) {
    return Status::OK();
  } else {
    return errors::FailedPrecondition(
        "Failed to close %d debug-events files associated with tfdbg",
        failed_to_close_files.size());
  }
}

// static
std::unordered_map<string, std::unique_ptr<DebugEventsWriter>>*
DebugEventsWriter::GetDebugEventsWriterMap() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_25(mht_25_v, 722, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::GetDebugEventsWriterMap");

  static std::unordered_map<string, std::unique_ptr<DebugEventsWriter>>*
      writer_pool =
          new std::unordered_map<string, std::unique_ptr<DebugEventsWriter>>();
  return writer_pool;
}

DebugEventsWriter::DebugEventsWriter(const string& dump_root,
                                     const string& tfdbg_run_id,
                                     int64_t circular_buffer_size)
    : env_(Env::Default()),
      dump_root_(dump_root),
      tfdbg_run_id_(tfdbg_run_id),
      is_initialized_(false),
      initialization_mu_(),
      circular_buffer_size_(circular_buffer_size),
      execution_buffer_(),
      execution_buffer_mu_(),
      graph_execution_trace_buffer_(),
      graph_execution_trace_buffer_mu_(),
      device_name_to_id_(),
      device_mu_() {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("dump_root: \"" + dump_root + "\"");
   mht_26_v.push_back("tfdbg_run_id: \"" + tfdbg_run_id + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_26(mht_26_v, 748, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::DebugEventsWriter");
}

Status DebugEventsWriter::InitNonMetadataFile(DebugEventFileType type) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_27(mht_27_v, 753, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::InitNonMetadataFile");

  std::unique_ptr<SingleDebugEventFileWriter>* writer = nullptr;
  SelectWriter(type, &writer);
  const string filename = GetFileNameInternal(type);
  writer->reset();

  writer->reset(new SingleDebugEventFileWriter(filename));
  if (*writer == nullptr) {
    return errors::Unknown("Could not create debug event file writer for ",
                           filename);
  }
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      (*writer)->Init(), "Initializing debug event writer at path ", filename);
  VLOG(1) << "Successfully opened debug event file: " << filename;

  return Status::OK();
}

Status DebugEventsWriter::SerializeAndWriteDebugEvent(DebugEvent* debug_event,
                                                      DebugEventFileType type) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_28(mht_28_v, 775, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::SerializeAndWriteDebugEvent");

  std::unique_ptr<SingleDebugEventFileWriter>* writer = nullptr;
  SelectWriter(type, &writer);
  if (writer != nullptr) {
    // Timestamp is in seconds, with double precision.
    MaybeSetDebugEventTimestamp(debug_event, env_);
    string str;
    debug_event->AppendToString(&str);
    (*writer)->WriteSerializedDebugEvent(str);
    return Status::OK();
  } else {
    return errors::Internal(
        "Unable to find debug events file writer for DebugEventsFileType ",
        type);
  }
}

void DebugEventsWriter::SelectWriter(
    DebugEventFileType type,
    std::unique_ptr<SingleDebugEventFileWriter>** writer) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_29(mht_29_v, 797, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::SelectWriter");

  switch (type) {
    case METADATA:
      *writer = &metadata_writer_;
      break;
    case SOURCE_FILES:
      *writer = &source_files_writer_;
      break;
    case STACK_FRAMES:
      *writer = &stack_frames_writer_;
      break;
    case GRAPHS:
      *writer = &graphs_writer_;
      break;
    case EXECUTION:
      *writer = &execution_writer_;
      break;
    case GRAPH_EXECUTION_TRACES:
      *writer = &graph_execution_traces_writer_;
      break;
  }
}

const string DebugEventsWriter::GetSuffix(DebugEventFileType type) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_30(mht_30_v, 823, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::GetSuffix");

  switch (type) {
    case METADATA:
      return kMetadataSuffix;
    case SOURCE_FILES:
      return kSourceFilesSuffix;
    case STACK_FRAMES:
      return kStackFramesSuffix;
    case GRAPHS:
      return kGraphsSuffix;
    case EXECUTION:
      return kExecutionSuffix;
    case GRAPH_EXECUTION_TRACES:
      return kGraphExecutionTracesSuffix;
    default:
      string suffix;
      return suffix;
  }
}

string DebugEventsWriter::GetFileNameInternal(DebugEventFileType type) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writerDTcc mht_31(mht_31_v, 846, "", "./tensorflow/core/util/debug_events_writer.cc", "DebugEventsWriter::GetFileNameInternal");

  const string suffix = GetSuffix(type);
  return strings::StrCat(file_prefix_, ".", suffix);
}

}  // namespace tfdbg
}  // namespace tensorflow
