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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc() {
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

#include "tensorflow/core/distributed_runtime/message_wrappers.h"

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"

namespace tensorflow {

bool ParseTensorProtoToTensor(const TensorProto& tensor_proto,
                              Tensor* out_tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ParseTensorProtoToTensor");

  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *out_tensor = parsed;
      return true;
    }
  }
  return false;
}

const string& InMemoryRunStepRequest::session_handle() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::session_handle");

  return session_handle_;
}

void InMemoryRunStepRequest::set_session_handle(const string& handle) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::set_session_handle");

  session_handle_ = handle;
}

const string& InMemoryRunStepRequest::partial_run_handle() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::partial_run_handle");

  return partial_run_handle_;
}

void InMemoryRunStepRequest::set_partial_run_handle(const string& handle) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::set_partial_run_handle");

  partial_run_handle_ = handle;
}

size_t InMemoryRunStepRequest::num_feeds() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_5(mht_5_v, 240, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::num_feeds");
 return feeds_.size(); }
const string& InMemoryRunStepRequest::feed_name(size_t i) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_6(mht_6_v, 244, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::feed_name");

  return feeds_[i].first;
}

Status InMemoryRunStepRequest::FeedValue(size_t i, Tensor* out_tensor) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_7(mht_7_v, 251, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::FeedValue");

  *out_tensor = feeds_[i].second;
  return Status::OK();
}

Status InMemoryRunStepRequest::FeedValue(size_t i,
                                         TensorProto* out_tensor) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_8(mht_8_v, 260, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::FeedValue");

  feeds_[i].second.AsProtoTensorContent(out_tensor);
  return Status::OK();
}

void InMemoryRunStepRequest::add_feed(const string& name, const Tensor& value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_9(mht_9_v, 269, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::add_feed");

  feeds_.emplace_back(name, value);
}

size_t InMemoryRunStepRequest::num_fetches() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_10(mht_10_v, 276, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::num_fetches");
 return fetches_.size(); }
const string& InMemoryRunStepRequest::fetch_name(size_t i) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_11(mht_11_v, 280, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::fetch_name");

  return fetches_[i];
}
void InMemoryRunStepRequest::add_fetch(const string& name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_12(mht_12_v, 287, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::add_fetch");

  fetches_.push_back(name);
}

size_t InMemoryRunStepRequest::num_targets() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_13(mht_13_v, 294, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::num_targets");
 return targets_.size(); }
const string& InMemoryRunStepRequest::target_name(size_t i) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_14(mht_14_v, 298, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::target_name");

  return targets_[i];
}
void InMemoryRunStepRequest::add_target(const string& name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_15(mht_15_v, 305, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::add_target");

  targets_.push_back(name);
}

const RunOptions& InMemoryRunStepRequest::options() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_16(mht_16_v, 312, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::options");
 return options_; }

RunOptions* InMemoryRunStepRequest::mutable_options() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_17(mht_17_v, 317, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::mutable_options");
 return &options_; }

bool InMemoryRunStepRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_18(mht_18_v, 322, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::store_errors_in_response_body");

  return store_errors_in_response_body_;
}

int64_t InMemoryRunStepRequest::request_id() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_19(mht_19_v, 329, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::request_id");

  return 0;  // no need to track request id for local version.
}

void InMemoryRunStepRequest::set_store_errors_in_response_body(
    bool store_errors) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_20(mht_20_v, 337, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::set_store_errors_in_response_body");

  store_errors_in_response_body_ = store_errors;
}

string InMemoryRunStepRequest::DebugString() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_21(mht_21_v, 344, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::DebugString");

  return ToProto().DebugString();
}

const RunStepRequest& InMemoryRunStepRequest::ToProto() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_22(mht_22_v, 351, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepRequest::ToProto");

  if (!proto_version_) {
    proto_version_.reset(new RunStepRequest);
    proto_version_->set_session_handle(session_handle());
    proto_version_->set_partial_run_handle(partial_run_handle());
    for (size_t i = 0; i < num_feeds(); ++i) {
      auto feed = proto_version_->add_feed();
      feed->set_name(feed_name(i));
      feeds_[i].second.AsProtoTensorContent(feed->mutable_tensor());
    }
    for (size_t i = 0; i < num_fetches(); ++i) {
      proto_version_->add_fetch(fetch_name(i));
    }
    for (size_t i = 0; i < num_targets(); ++i) {
      proto_version_->add_target(target_name(i));
    }
    *proto_version_->mutable_options() = options();
  }
  return *proto_version_;
}

const string& MutableProtoRunStepRequest::session_handle() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_23(mht_23_v, 375, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::session_handle");

  return request_.session_handle();
}
void MutableProtoRunStepRequest::set_session_handle(const string& handle) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_24(mht_24_v, 382, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::set_session_handle");

  request_.set_session_handle(handle);
}

const string& MutableProtoRunStepRequest::partial_run_handle() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_25(mht_25_v, 389, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::partial_run_handle");

  return request_.partial_run_handle();
}
void MutableProtoRunStepRequest::set_partial_run_handle(const string& handle) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_26(mht_26_v, 396, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::set_partial_run_handle");

  request_.set_partial_run_handle(handle);
}

size_t MutableProtoRunStepRequest::num_feeds() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_27(mht_27_v, 403, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::num_feeds");

  return request_.feed_size();
}
const string& MutableProtoRunStepRequest::feed_name(size_t i) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_28(mht_28_v, 409, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::feed_name");

  return request_.feed(i).name();
}
Status MutableProtoRunStepRequest::FeedValue(size_t i,
                                             Tensor* out_tensor) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_29(mht_29_v, 416, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::FeedValue");

  if (!ParseTensorProtoToTensor(request_.feed(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status MutableProtoRunStepRequest::FeedValue(size_t i,
                                             TensorProto* out_tensor) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_30(mht_30_v, 428, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::FeedValue");

  *out_tensor = request_.feed(i).tensor();
  return Status::OK();
}

void MutableProtoRunStepRequest::add_feed(const string& name,
                                          const Tensor& value) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_31(mht_31_v, 438, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::add_feed");

  NamedTensorProto* feed = request_.add_feed();
  feed->set_name(name);
  TensorProto* value_proto = feed->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

size_t MutableProtoRunStepRequest::num_fetches() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_32(mht_32_v, 448, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::num_fetches");

  return request_.fetch_size();
}

const string& MutableProtoRunStepRequest::fetch_name(size_t i) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_33(mht_33_v, 455, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::fetch_name");

  return request_.fetch(i);
}
void MutableProtoRunStepRequest::add_fetch(const string& name) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_34(mht_34_v, 462, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::add_fetch");

  request_.add_fetch(name);
}

size_t MutableProtoRunStepRequest::num_targets() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_35(mht_35_v, 469, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::num_targets");

  return request_.target_size();
}

const string& MutableProtoRunStepRequest::target_name(size_t i) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_36(mht_36_v, 476, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::target_name");

  return request_.target(i);
}

void MutableProtoRunStepRequest::add_target(const string& name) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_37(mht_37_v, 484, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::add_target");

  request_.add_target(name);
}

const RunOptions& MutableProtoRunStepRequest::options() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_38(mht_38_v, 491, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::options");

  return request_.options();
}

RunOptions* MutableProtoRunStepRequest::mutable_options() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_39(mht_39_v, 498, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::mutable_options");

  return request_.mutable_options();
}

bool MutableProtoRunStepRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_40(mht_40_v, 505, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::store_errors_in_response_body");

  return request_.store_errors_in_response_body();
}

void MutableProtoRunStepRequest::set_store_errors_in_response_body(
    bool store_errors) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_41(mht_41_v, 513, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::set_store_errors_in_response_body");

  request_.set_store_errors_in_response_body(store_errors);
}

int64_t MutableProtoRunStepRequest::request_id() const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_42(mht_42_v, 520, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::request_id");

  return request_.request_id();
}

string MutableProtoRunStepRequest::DebugString() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_43(mht_43_v, 527, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::DebugString");

  return request_.DebugString();
}

const RunStepRequest& MutableProtoRunStepRequest::ToProto() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_44(mht_44_v, 534, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunStepRequest::ToProto");

  return request_;
}

ProtoRunStepRequest::ProtoRunStepRequest(const RunStepRequest* request)
    : request_(request) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_45(mht_45_v, 542, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::ProtoRunStepRequest");
}

const string& ProtoRunStepRequest::session_handle() const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_46(mht_46_v, 547, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::session_handle");

  return request_->session_handle();
}

const string& ProtoRunStepRequest::partial_run_handle() const {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_47(mht_47_v, 554, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::partial_run_handle");

  return request_->partial_run_handle();
}

size_t ProtoRunStepRequest::num_feeds() const {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_48(mht_48_v, 561, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::num_feeds");
 return request_->feed_size(); }

const string& ProtoRunStepRequest::feed_name(size_t i) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_49(mht_49_v, 566, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::feed_name");

  return request_->feed(i).name();
}

Status ProtoRunStepRequest::FeedValue(size_t i, Tensor* out_tensor) const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_50(mht_50_v, 573, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::FeedValue");

  if (!ParseTensorProtoToTensor(request_->feed(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status ProtoRunStepRequest::FeedValue(size_t i, TensorProto* out_tensor) const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_51(mht_51_v, 584, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::FeedValue");

  *out_tensor = request_->feed(i).tensor();
  return Status::OK();
}

size_t ProtoRunStepRequest::num_fetches() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_52(mht_52_v, 592, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::num_fetches");

  return request_->fetch_size();
}

const string& ProtoRunStepRequest::fetch_name(size_t i) const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_53(mht_53_v, 599, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::fetch_name");

  return request_->fetch(i);
}

size_t ProtoRunStepRequest::num_targets() const {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_54(mht_54_v, 606, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::num_targets");

  return request_->target_size();
}

const string& ProtoRunStepRequest::target_name(size_t i) const {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_55(mht_55_v, 613, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::target_name");

  return request_->target(i);
}

const RunOptions& ProtoRunStepRequest::options() const {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_56(mht_56_v, 620, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::options");

  return request_->options();
}

bool ProtoRunStepRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_57(mht_57_v, 627, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::store_errors_in_response_body");

  return request_->store_errors_in_response_body();
}

int64_t ProtoRunStepRequest::request_id() const {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_58(mht_58_v, 634, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::request_id");

  return request_->request_id();
}

string ProtoRunStepRequest::DebugString() const {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_59(mht_59_v, 641, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::DebugString");

  return request_->DebugString();
}

const RunStepRequest& ProtoRunStepRequest::ToProto() const {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_60(mht_60_v, 648, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunStepRequest::ToProto");
 return *request_; }

const string& InMemoryRunGraphRequest::session_handle() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_61(mht_61_v, 653, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::session_handle");

  return session_handle_;
}

bool InMemoryRunGraphRequest::create_worker_session_called() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_62(mht_62_v, 660, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::create_worker_session_called");

  return create_worker_session_called_;
}

void InMemoryRunGraphRequest::set_session_handle(const string& handle) {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_63(mht_63_v, 668, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_session_handle");

  session_handle_ = handle;
}

void InMemoryRunGraphRequest::set_create_worker_session_called(bool called) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_64(mht_64_v, 675, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_create_worker_session_called");

  create_worker_session_called_ = called;
}

const string& InMemoryRunGraphRequest::graph_handle() const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_65(mht_65_v, 682, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::graph_handle");

  return graph_handle_;
}

void InMemoryRunGraphRequest::set_graph_handle(const string& handle) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_66(mht_66_v, 690, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_graph_handle");

  graph_handle_ = handle;
}

int64_t InMemoryRunGraphRequest::step_id() const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_67(mht_67_v, 697, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::step_id");
 return step_id_; }

void InMemoryRunGraphRequest::set_step_id(int64_t step_id) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_68(mht_68_v, 702, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_step_id");

  step_id_ = step_id;
}

const ExecutorOpts& InMemoryRunGraphRequest::exec_opts() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_69(mht_69_v, 709, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::exec_opts");

  return exec_opts_;
}

ExecutorOpts* InMemoryRunGraphRequest::mutable_exec_opts() {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_70(mht_70_v, 716, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::mutable_exec_opts");

  return &exec_opts_;
}

size_t InMemoryRunGraphRequest::num_sends() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_71(mht_71_v, 723, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::num_sends");
 return sends_.size(); }

const string& InMemoryRunGraphRequest::send_key(size_t i) const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_72(mht_72_v, 728, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::send_key");

  return sends_[i].first;
}

Status InMemoryRunGraphRequest::SendValue(size_t i, Tensor* out_tensor) const {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_73(mht_73_v, 735, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::SendValue");

  *out_tensor = sends_[i].second;
  return Status::OK();
}

Status InMemoryRunGraphRequest::AddSendFromRunStepRequest(
    const RunStepRequestWrapper& run_step_request, size_t i,
    const string& send_key) {
   std::vector<std::string> mht_74_v;
   mht_74_v.push_back("send_key: \"" + send_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_74(mht_74_v, 746, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::AddSendFromRunStepRequest");

  Tensor tensor;
  TF_RETURN_IF_ERROR(run_step_request.FeedValue(i, &tensor));
  sends_.emplace_back(send_key, std::move(tensor));
  return Status::OK();
}

Status InMemoryRunGraphRequest::AddSendFromRunCallableRequest(
    const RunCallableRequest& run_callable_request, size_t i,
    const string& send_key) {
   std::vector<std::string> mht_75_v;
   mht_75_v.push_back("send_key: \"" + send_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_75(mht_75_v, 759, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::AddSendFromRunCallableRequest");

  Tensor tensor;
  if (!ParseTensorProtoToTensor(run_callable_request.feed(i), &tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  }
  sends_.emplace_back(send_key, std::move(tensor));
  return Status::OK();
}

size_t InMemoryRunGraphRequest::num_recvs() const {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_76(mht_76_v, 771, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::num_recvs");
 return recvs_.size(); }

const string& InMemoryRunGraphRequest::recv_key(size_t i) const {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_77(mht_77_v, 776, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::recv_key");

  return recvs_[i];
}

void InMemoryRunGraphRequest::add_recv_key(const string& recv_key) {
   std::vector<std::string> mht_78_v;
   mht_78_v.push_back("recv_key: \"" + recv_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_78(mht_78_v, 784, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::add_recv_key");

  recvs_.push_back(recv_key);
}

bool InMemoryRunGraphRequest::is_partial() const {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_79(mht_79_v, 791, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::is_partial");
 return is_partial_; }

void InMemoryRunGraphRequest::set_is_partial(bool is_partial) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_80(mht_80_v, 796, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_is_partial");

  is_partial_ = is_partial;
}

bool InMemoryRunGraphRequest::is_last_partial_run() const {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_81(mht_81_v, 803, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::is_last_partial_run");

  return is_last_partial_run_;
}

void InMemoryRunGraphRequest::set_is_last_partial_run(
    bool is_last_partial_run) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_82(mht_82_v, 811, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_is_last_partial_run");

  is_last_partial_run_ = is_last_partial_run;
}

bool InMemoryRunGraphRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_83(mht_83_v, 818, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::store_errors_in_response_body");

  return store_errors_in_response_body_;
}

void InMemoryRunGraphRequest::set_store_errors_in_response_body(
    bool store_errors) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_84(mht_84_v, 826, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_store_errors_in_response_body");

  store_errors_in_response_body_ = store_errors;
}

int64_t InMemoryRunGraphRequest::request_id() const {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_85(mht_85_v, 833, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::request_id");
 return request_id_; }

void InMemoryRunGraphRequest::set_request_id(int64_t request_id) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_86(mht_86_v, 838, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::set_request_id");

  request_id_ = request_id;
}

const RunGraphRequest& InMemoryRunGraphRequest::ToProto() const {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_87(mht_87_v, 845, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphRequest::ToProto");

  if (!proto_version_) {
    proto_version_.reset(new RunGraphRequest);
    proto_version_->set_session_handle(session_handle());
    proto_version_->set_create_worker_session_called(
        create_worker_session_called());
    proto_version_->set_graph_handle(graph_handle());
    proto_version_->set_step_id(step_id());
    *proto_version_->mutable_exec_opts() = exec_opts();
    for (size_t i = 0; i < num_sends(); ++i) {
      auto send = proto_version_->add_send();
      send->set_name(send_key(i));
      sends_[i].second.AsProtoTensorContent(send->mutable_tensor());
    }
    for (size_t i = 0; i < num_recvs(); ++i) {
      proto_version_->add_recv_key(recv_key(i));
    }
    proto_version_->set_is_partial(is_partial());
    proto_version_->set_is_last_partial_run(is_last_partial_run());
  }
  proto_version_->set_store_errors_in_response_body(
      store_errors_in_response_body_);
  proto_version_->set_request_id(request_id_);
  return *proto_version_;
}

const string& MutableProtoRunGraphRequest::session_handle() const {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_88(mht_88_v, 874, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::session_handle");

  return request_.session_handle();
}

void MutableProtoRunGraphRequest::set_session_handle(const string& handle) {
   std::vector<std::string> mht_89_v;
   mht_89_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_89(mht_89_v, 882, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_session_handle");

  request_.set_session_handle(handle);
}

bool MutableProtoRunGraphRequest::create_worker_session_called() const {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_90(mht_90_v, 889, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::create_worker_session_called");

  return request_.create_worker_session_called();
}

void MutableProtoRunGraphRequest::set_create_worker_session_called(
    bool called) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_91(mht_91_v, 897, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_create_worker_session_called");

  request_.set_create_worker_session_called(called);
}

const string& MutableProtoRunGraphRequest::graph_handle() const {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_92(mht_92_v, 904, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::graph_handle");

  return request_.graph_handle();
}

void MutableProtoRunGraphRequest::set_graph_handle(const string& handle) {
   std::vector<std::string> mht_93_v;
   mht_93_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_93(mht_93_v, 912, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_graph_handle");

  request_.set_graph_handle(handle);
}

int64_t MutableProtoRunGraphRequest::step_id() const {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_94(mht_94_v, 919, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::step_id");

  return request_.step_id();
}

void MutableProtoRunGraphRequest::set_step_id(int64_t step_id) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_95(mht_95_v, 926, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_step_id");

  request_.set_step_id(step_id);
}

const ExecutorOpts& MutableProtoRunGraphRequest::exec_opts() const {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_96(mht_96_v, 933, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::exec_opts");

  return request_.exec_opts();
}

ExecutorOpts* MutableProtoRunGraphRequest::mutable_exec_opts() {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_97(mht_97_v, 940, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::mutable_exec_opts");

  return request_.mutable_exec_opts();
}

size_t MutableProtoRunGraphRequest::num_sends() const {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_98(mht_98_v, 947, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::num_sends");

  return request_.send_size();
}

const string& MutableProtoRunGraphRequest::send_key(size_t i) const {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_99(mht_99_v, 954, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::send_key");

  return request_.send(i).name();
}

Status MutableProtoRunGraphRequest::SendValue(size_t i,
                                              Tensor* out_tensor) const {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_100(mht_100_v, 962, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::SendValue");

  if (!ParseTensorProtoToTensor(request_.send(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status MutableProtoRunGraphRequest::AddSendFromRunStepRequest(
    const RunStepRequestWrapper& run_step_request, size_t i,
    const string& send_key) {
   std::vector<std::string> mht_101_v;
   mht_101_v.push_back("send_key: \"" + send_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_101(mht_101_v, 976, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::AddSendFromRunStepRequest");

  NamedTensorProto* send = request_.add_send();
  send->set_name(send_key);
  TF_RETURN_IF_ERROR(run_step_request.FeedValue(i, send->mutable_tensor()));
  return Status::OK();
}

Status MutableProtoRunGraphRequest::AddSendFromRunCallableRequest(
    const RunCallableRequest& run_callable_request, size_t i,
    const string& send_key) {
   std::vector<std::string> mht_102_v;
   mht_102_v.push_back("send_key: \"" + send_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_102(mht_102_v, 989, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::AddSendFromRunCallableRequest");

  NamedTensorProto* send = request_.add_send();
  send->set_name(send_key);
  *send->mutable_tensor() = run_callable_request.feed(i);
  return Status::OK();
}

size_t MutableProtoRunGraphRequest::num_recvs() const {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_103(mht_103_v, 999, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::num_recvs");

  return request_.recv_key_size();
}

const string& MutableProtoRunGraphRequest::recv_key(size_t i) const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_104(mht_104_v, 1006, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::recv_key");

  return request_.recv_key(i);
}

void MutableProtoRunGraphRequest::add_recv_key(const string& recv_key) {
   std::vector<std::string> mht_105_v;
   mht_105_v.push_back("recv_key: \"" + recv_key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_105(mht_105_v, 1014, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::add_recv_key");

  request_.add_recv_key(recv_key);
}

bool MutableProtoRunGraphRequest::is_partial() const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_106(mht_106_v, 1021, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::is_partial");

  return request_.is_partial();
}

void MutableProtoRunGraphRequest::set_is_partial(bool is_partial) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_107(mht_107_v, 1028, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_is_partial");

  request_.set_is_partial(is_partial);
}

bool MutableProtoRunGraphRequest::is_last_partial_run() const {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_108(mht_108_v, 1035, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::is_last_partial_run");

  return request_.is_last_partial_run();
}

void MutableProtoRunGraphRequest::set_is_last_partial_run(
    bool is_last_partial_run) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_109(mht_109_v, 1043, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_is_last_partial_run");

  request_.set_is_last_partial_run(is_last_partial_run);
}

bool MutableProtoRunGraphRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_110(mht_110_v, 1050, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::store_errors_in_response_body");

  return request_.store_errors_in_response_body();
}

void MutableProtoRunGraphRequest::set_store_errors_in_response_body(
    bool store_errors) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_111(mht_111_v, 1058, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_store_errors_in_response_body");

  request_.set_store_errors_in_response_body(store_errors);
}

int64_t MutableProtoRunGraphRequest::request_id() const {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_112(mht_112_v, 1065, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::request_id");

  return request_.request_id();
}

void MutableProtoRunGraphRequest::set_request_id(int64_t request_id) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_113(mht_113_v, 1072, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::set_request_id");

  request_.set_request_id(request_id);
}

const RunGraphRequest& MutableProtoRunGraphRequest::ToProto() const {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_114(mht_114_v, 1079, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableProtoRunGraphRequest::ToProto");

  return request_;
}

ProtoRunGraphRequest::ProtoRunGraphRequest(const RunGraphRequest* request)
    : request_(request) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_115(mht_115_v, 1087, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::ProtoRunGraphRequest");
}

const string& ProtoRunGraphRequest::session_handle() const {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_116(mht_116_v, 1092, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::session_handle");

  return request_->session_handle();
}

bool ProtoRunGraphRequest::create_worker_session_called() const {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_117(mht_117_v, 1099, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::create_worker_session_called");

  return request_->create_worker_session_called();
}

const string& ProtoRunGraphRequest::graph_handle() const {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_118(mht_118_v, 1106, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::graph_handle");

  return request_->graph_handle();
}

int64_t ProtoRunGraphRequest::step_id() const {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_119(mht_119_v, 1113, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::step_id");
 return request_->step_id(); }

const ExecutorOpts& ProtoRunGraphRequest::exec_opts() const {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_120(mht_120_v, 1118, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::exec_opts");

  return request_->exec_opts();
}

size_t ProtoRunGraphRequest::num_sends() const {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_121(mht_121_v, 1125, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::num_sends");
 return request_->send_size(); }

const string& ProtoRunGraphRequest::send_key(size_t i) const {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_122(mht_122_v, 1130, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::send_key");

  return request_->send(i).name();
}

Status ProtoRunGraphRequest::SendValue(size_t i, Tensor* out_tensor) const {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_123(mht_123_v, 1137, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::SendValue");

  if (!ParseTensorProtoToTensor(request_->send(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

size_t ProtoRunGraphRequest::num_recvs() const {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_124(mht_124_v, 1148, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::num_recvs");

  return request_->recv_key_size();
}

const string& ProtoRunGraphRequest::recv_key(size_t i) const {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_125(mht_125_v, 1155, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::recv_key");

  return request_->recv_key(i);
}

bool ProtoRunGraphRequest::is_partial() const {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_126(mht_126_v, 1162, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::is_partial");
 return request_->is_partial(); }

bool ProtoRunGraphRequest::is_last_partial_run() const {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_127(mht_127_v, 1167, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::is_last_partial_run");

  return request_->is_last_partial_run();
}

bool ProtoRunGraphRequest::store_errors_in_response_body() const {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_128(mht_128_v, 1174, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::store_errors_in_response_body");

  return request_->store_errors_in_response_body();
}

int64_t ProtoRunGraphRequest::request_id() const {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_129(mht_129_v, 1181, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::request_id");

  return request_->request_id();
}

const RunGraphRequest& ProtoRunGraphRequest::ToProto() const {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_130(mht_130_v, 1188, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "ProtoRunGraphRequest::ToProto");

  return *request_;
}

size_t InMemoryRunGraphResponse::num_recvs() const {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_131(mht_131_v, 1195, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::num_recvs");
 return recvs_.size(); }

const string& InMemoryRunGraphResponse::recv_key(size_t i) const {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_132(mht_132_v, 1200, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::recv_key");

  return recvs_[i].first;
}

Status InMemoryRunGraphResponse::RecvValue(size_t i, TensorProto* out_tensor) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_133(mht_133_v, 1207, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::RecvValue");

  recvs_[i].second.AsProtoTensorContent(out_tensor);
  return Status::OK();
}

Status InMemoryRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_134(mht_134_v, 1215, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::RecvValue");

  *out_tensor = recvs_[i].second;
  return Status::OK();
}

void InMemoryRunGraphResponse::AddRecv(const string& key, const Tensor& value) {
   std::vector<std::string> mht_135_v;
   mht_135_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_135(mht_135_v, 1224, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::AddRecv");

  recvs_.emplace_back(key, value);
}

StepStats* InMemoryRunGraphResponse::mutable_step_stats() {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_136(mht_136_v, 1231, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::mutable_step_stats");

  return &step_stats_;
}

CostGraphDef* InMemoryRunGraphResponse::mutable_cost_graph() {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_137(mht_137_v, 1238, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::mutable_cost_graph");

  return &cost_graph_;
}

Status InMemoryRunGraphResponse::status() const {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_138(mht_138_v, 1245, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::status");
 return status_; }

errors::Code InMemoryRunGraphResponse::status_code() const {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_139(mht_139_v, 1250, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::status_code");

  return status_.code();
}

const string& InMemoryRunGraphResponse::status_error_message() const {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_140(mht_140_v, 1257, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::status_error_message");

  return status_.error_message();
}

void InMemoryRunGraphResponse::set_status(const Status& status) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_141(mht_141_v, 1264, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::set_status");

  status_ = status;
}

RunGraphResponse* InMemoryRunGraphResponse::get_proto() {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_142(mht_142_v, 1271, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::get_proto");

  LOG(FATAL) << "Cannot get a mutable protobuf for an InMemoryRunGraphResponse";
  return nullptr;
}

size_t InMemoryRunGraphResponse::num_partition_graphs() const {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_143(mht_143_v, 1279, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::num_partition_graphs");

  return partition_graphs_.size();
}

GraphDef* InMemoryRunGraphResponse::mutable_partition_graph(size_t i) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_144(mht_144_v, 1286, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::mutable_partition_graph");

  return &partition_graphs_[i];
}

void InMemoryRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_145(mht_145_v, 1294, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunGraphResponse::AddPartitionGraph");

  partition_graphs_.push_back(partition_graph);
}

size_t OwnedProtoRunGraphResponse::num_recvs() const {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_146(mht_146_v, 1301, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::num_recvs");

  return response_.recv_size();
}

const string& OwnedProtoRunGraphResponse::recv_key(size_t i) const {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_147(mht_147_v, 1308, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::recv_key");

  return response_.recv(i).name();
}

Status OwnedProtoRunGraphResponse::RecvValue(size_t i,
                                             TensorProto* out_tensor) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_148(mht_148_v, 1316, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::RecvValue");

  out_tensor->Swap(response_.mutable_recv(i)->mutable_tensor());
  return Status::OK();
}

Status OwnedProtoRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_149(mht_149_v, 1324, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::RecvValue");

  if (!ParseTensorProtoToTensor(response_.recv(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for recv value ", i);
  } else {
    return Status::OK();
  }
}

void OwnedProtoRunGraphResponse::AddRecv(const string& key,
                                         const Tensor& value) {
   std::vector<std::string> mht_150_v;
   mht_150_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_150(mht_150_v, 1337, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::AddRecv");

  NamedTensorProto* recv = response_.add_recv();
  recv->set_name(key);
  TensorProto* value_proto = recv->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

StepStats* OwnedProtoRunGraphResponse::mutable_step_stats() {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_151(mht_151_v, 1347, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::mutable_step_stats");

  return response_.mutable_step_stats();
}

CostGraphDef* OwnedProtoRunGraphResponse::mutable_cost_graph() {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_152(mht_152_v, 1354, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::mutable_cost_graph");

  return response_.mutable_cost_graph();
}

Status OwnedProtoRunGraphResponse::status() const {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_153(mht_153_v, 1361, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::status");

  return Status(response_.status_code(), response_.status_error_message());
}

errors::Code OwnedProtoRunGraphResponse::status_code() const {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_154(mht_154_v, 1368, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::status_code");

  return response_.status_code();
}

const string& OwnedProtoRunGraphResponse::status_error_message() const {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_155(mht_155_v, 1375, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::status_error_message");

  return response_.status_error_message();
}

void OwnedProtoRunGraphResponse::set_status(const Status& status) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_156(mht_156_v, 1382, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::set_status");

  response_.set_status_code(status.code());
  response_.set_status_error_message(status.error_message());
}

RunGraphResponse* OwnedProtoRunGraphResponse::get_proto() {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_157(mht_157_v, 1390, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::get_proto");
 return &response_; }

size_t OwnedProtoRunGraphResponse::num_partition_graphs() const {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_158(mht_158_v, 1395, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::num_partition_graphs");

  return response_.partition_graph_size();
}

GraphDef* OwnedProtoRunGraphResponse::mutable_partition_graph(size_t i) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_159(mht_159_v, 1402, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::mutable_partition_graph");

  return response_.mutable_partition_graph(i);
}

void OwnedProtoRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_160(mht_160_v, 1410, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunGraphResponse::AddPartitionGraph");

  GraphDef* graph_def = response_.mutable_partition_graph()->Add();
  *graph_def = partition_graph;
}

NonOwnedProtoRunGraphResponse::NonOwnedProtoRunGraphResponse(
    RunGraphResponse* response)
    : response_(response) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_161(mht_161_v, 1420, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::NonOwnedProtoRunGraphResponse");
}

size_t NonOwnedProtoRunGraphResponse::num_recvs() const {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_162(mht_162_v, 1425, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::num_recvs");

  return response_->recv_size();
}

const string& NonOwnedProtoRunGraphResponse::recv_key(size_t i) const {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_163(mht_163_v, 1432, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::recv_key");

  return response_->recv(i).name();
}

Status NonOwnedProtoRunGraphResponse::RecvValue(size_t i,
                                                TensorProto* out_tensor) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_164(mht_164_v, 1440, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::RecvValue");

  out_tensor->Swap(response_->mutable_recv(i)->mutable_tensor());
  return Status::OK();
}

Status NonOwnedProtoRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_165(mht_165_v, 1448, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::RecvValue");

  if (!ParseTensorProtoToTensor(response_->recv(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for recv value ", i);
  } else {
    return Status::OK();
  }
}

void NonOwnedProtoRunGraphResponse::AddRecv(const string& key,
                                            const Tensor& value) {
   std::vector<std::string> mht_166_v;
   mht_166_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_166(mht_166_v, 1461, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::AddRecv");

  NamedTensorProto* recv = response_->add_recv();
  recv->set_name(key);
  TensorProto* value_proto = recv->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

StepStats* NonOwnedProtoRunGraphResponse::mutable_step_stats() {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_167(mht_167_v, 1471, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::mutable_step_stats");

  return response_->mutable_step_stats();
}

CostGraphDef* NonOwnedProtoRunGraphResponse::mutable_cost_graph() {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_168(mht_168_v, 1478, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::mutable_cost_graph");

  return response_->mutable_cost_graph();
}

Status NonOwnedProtoRunGraphResponse::status() const {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_169(mht_169_v, 1485, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::status");

  return Status(response_->status_code(), response_->status_error_message());
}

errors::Code NonOwnedProtoRunGraphResponse::status_code() const {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_170(mht_170_v, 1492, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::status_code");

  return response_->status_code();
}

const string& NonOwnedProtoRunGraphResponse::status_error_message() const {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_171(mht_171_v, 1499, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::status_error_message");

  return response_->status_error_message();
}

void NonOwnedProtoRunGraphResponse::set_status(const Status& status) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_172(mht_172_v, 1506, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::set_status");

  response_->set_status_code(status.code());
  response_->set_status_error_message(status.error_message());
}

RunGraphResponse* NonOwnedProtoRunGraphResponse::get_proto() {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_173(mht_173_v, 1514, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::get_proto");

  return response_;
}

size_t NonOwnedProtoRunGraphResponse::num_partition_graphs() const {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_174(mht_174_v, 1521, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::num_partition_graphs");

  return response_->partition_graph_size();
}

GraphDef* NonOwnedProtoRunGraphResponse::mutable_partition_graph(size_t i) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_175(mht_175_v, 1528, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::mutable_partition_graph");

  return response_->mutable_partition_graph(i);
}

void NonOwnedProtoRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_176(mht_176_v, 1536, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunGraphResponse::AddPartitionGraph");

  GraphDef* graph_def = response_->add_partition_graph();
  *graph_def = partition_graph;
}

MutableRunStepResponseWrapper::~MutableRunStepResponseWrapper() {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_177(mht_177_v, 1544, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "MutableRunStepResponseWrapper::~MutableRunStepResponseWrapper");
}

size_t InMemoryRunStepResponse::num_tensors() const {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_178(mht_178_v, 1549, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::num_tensors");
 return tensors_.size(); }

const string& InMemoryRunStepResponse::tensor_name(size_t i) const {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_179(mht_179_v, 1554, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::tensor_name");

  return tensors_[i].first;
}

Status InMemoryRunStepResponse::TensorValue(size_t i,
                                            Tensor* out_tensor) const {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_180(mht_180_v, 1562, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::TensorValue");

  *out_tensor = tensors_[i].second;
  return Status::OK();
}

const RunMetadata& InMemoryRunStepResponse::metadata() const {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_181(mht_181_v, 1570, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::metadata");

  return metadata_;
}

Status InMemoryRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* wrapper, size_t i) {
   std::vector<std::string> mht_182_v;
   mht_182_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_182(mht_182_v, 1579, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::AddTensorFromRunGraphResponse");

  Tensor tensor;
  TF_RETURN_IF_ERROR(wrapper->RecvValue(i, &tensor));
  tensors_.emplace_back(name, tensor);
  return Status::OK();
}

RunMetadata* InMemoryRunStepResponse::mutable_metadata() {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_183(mht_183_v, 1589, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::mutable_metadata");
 return &metadata_; }

Status InMemoryRunStepResponse::status() const {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_184(mht_184_v, 1594, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::status");
 return status_; }

errors::Code InMemoryRunStepResponse::status_code() const {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_185(mht_185_v, 1599, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::status_code");

  return status_.code();
}

const string& InMemoryRunStepResponse::status_error_message() const {
   std::vector<std::string> mht_186_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_186(mht_186_v, 1606, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::status_error_message");

  return status_.error_message();
}

void InMemoryRunStepResponse::set_status(const Status& status) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_187(mht_187_v, 1613, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::set_status");

  status_ = status;
}

RunStepResponse* InMemoryRunStepResponse::get_proto() {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_188(mht_188_v, 1620, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "InMemoryRunStepResponse::get_proto");

  LOG(FATAL) << "Cannot get a mutable protobuf for an InMemoryRunStepResponse";
  return nullptr;
}

size_t OwnedProtoRunStepResponse::num_tensors() const {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_189(mht_189_v, 1628, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::num_tensors");

  return response_.tensor_size();
}

const string& OwnedProtoRunStepResponse::tensor_name(size_t i) const {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_190(mht_190_v, 1635, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::tensor_name");

  return response_.tensor(i).name();
}

Status OwnedProtoRunStepResponse::TensorValue(size_t i,
                                              Tensor* out_tensor) const {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_191(mht_191_v, 1643, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::TensorValue");

  if (!ParseTensorProtoToTensor(response_.tensor(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for fetch value ", i);
  } else {
    return Status::OK();
  }
}

const RunMetadata& OwnedProtoRunStepResponse::metadata() const {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_192(mht_192_v, 1654, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::metadata");

  return response_.metadata();
}

Status OwnedProtoRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* run_graph_response,
    size_t i) {
   std::vector<std::string> mht_193_v;
   mht_193_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_193(mht_193_v, 1664, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::AddTensorFromRunGraphResponse");

  NamedTensorProto* response_tensor = response_.add_tensor();
  response_tensor->set_name(name);
  return run_graph_response->RecvValue(i, response_tensor->mutable_tensor());
}

RunMetadata* OwnedProtoRunStepResponse::mutable_metadata() {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_194(mht_194_v, 1673, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::mutable_metadata");

  return response_.mutable_metadata();
}

Status OwnedProtoRunStepResponse::status() const {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_195(mht_195_v, 1680, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::status");

  return Status(response_.status_code(), response_.status_error_message());
}

errors::Code OwnedProtoRunStepResponse::status_code() const {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_196(mht_196_v, 1687, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::status_code");

  return response_.status_code();
}

const string& OwnedProtoRunStepResponse::status_error_message() const {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_197(mht_197_v, 1694, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::status_error_message");

  return response_.status_error_message();
}

void OwnedProtoRunStepResponse::set_status(const Status& status) {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_198(mht_198_v, 1701, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::set_status");

  response_.set_status_code(status.code());
  response_.set_status_error_message(status.error_message());
}

RunStepResponse* OwnedProtoRunStepResponse::get_proto() {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_199(mht_199_v, 1709, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "OwnedProtoRunStepResponse::get_proto");
 return &response_; }

NonOwnedProtoRunStepResponse::NonOwnedProtoRunStepResponse(
    RunStepResponse* response)
    : response_(response) {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_200(mht_200_v, 1716, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::NonOwnedProtoRunStepResponse");
}

size_t NonOwnedProtoRunStepResponse::num_tensors() const {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_201(mht_201_v, 1721, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::num_tensors");

  return response_->tensor_size();
}

const string& NonOwnedProtoRunStepResponse::tensor_name(size_t i) const {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_202(mht_202_v, 1728, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::tensor_name");

  return response_->tensor(i).name();
}

Status NonOwnedProtoRunStepResponse::TensorValue(size_t i,
                                                 Tensor* out_tensor) const {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_203(mht_203_v, 1736, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::TensorValue");

  if (!ParseTensorProtoToTensor(response_->tensor(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for fetch value ", i);
  } else {
    return Status::OK();
  }
}

const RunMetadata& NonOwnedProtoRunStepResponse::metadata() const {
   std::vector<std::string> mht_204_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_204(mht_204_v, 1747, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::metadata");

  return response_->metadata();
}

Status NonOwnedProtoRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* run_graph_response,
    size_t i) {
   std::vector<std::string> mht_205_v;
   mht_205_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_205(mht_205_v, 1757, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::AddTensorFromRunGraphResponse");

  NamedTensorProto* response_tensor = response_->add_tensor();
  response_tensor->set_name(name);
  return run_graph_response->RecvValue(i, response_tensor->mutable_tensor());
}

RunMetadata* NonOwnedProtoRunStepResponse::mutable_metadata() {
   std::vector<std::string> mht_206_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_206(mht_206_v, 1766, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::mutable_metadata");

  return response_->mutable_metadata();
}

Status NonOwnedProtoRunStepResponse::status() const {
   std::vector<std::string> mht_207_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_207(mht_207_v, 1773, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::status");

  return Status(response_->status_code(), response_->status_error_message());
}

errors::Code NonOwnedProtoRunStepResponse::status_code() const {
   std::vector<std::string> mht_208_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_208(mht_208_v, 1780, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::status_code");

  return response_->status_code();
}

const string& NonOwnedProtoRunStepResponse::status_error_message() const {
   std::vector<std::string> mht_209_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_209(mht_209_v, 1787, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::status_error_message");

  return response_->status_error_message();
}

void NonOwnedProtoRunStepResponse::set_status(const Status& status) {
   std::vector<std::string> mht_210_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_210(mht_210_v, 1794, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::set_status");

  response_->set_status_code(status.code());
  response_->set_status_error_message(status.error_message());
}

RunStepResponse* NonOwnedProtoRunStepResponse::get_proto() {
   std::vector<std::string> mht_211_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappersDTcc mht_211(mht_211_v, 1802, "", "./tensorflow/core/distributed_runtime/message_wrappers.cc", "NonOwnedProtoRunStepResponse::get_proto");
 return response_; }

}  // namespace tensorflow
