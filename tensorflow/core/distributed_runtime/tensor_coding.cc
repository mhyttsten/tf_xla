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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/tensor_coding.h"

#include "google/protobuf/any.pb.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {

TensorResponse::Source::~Source() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::Source::~Source");
}

void TensorResponse::Clear() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::Clear");

  on_host_ = false;
  device_ = nullptr;
  alloc_attrs_ = AllocatorAttributes();
  allocator_ = nullptr;
  already_used_ = false;
  ClearTensor();
}

void TensorResponse::ClearTensor() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::ClearTensor");

  meta_.Clear();
  tensor_ = Tensor();
}

void TensorResponse::InitAlloc(DeviceBase* d, const AllocatorAttributes& aa) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::InitAlloc");

  Clear();
  device_ = d;
  alloc_attrs_ = aa;
  const DeviceAttributes& da = d->attributes();
  if (alloc_attrs_.on_host() || da.device_type() == "CPU") {
    on_host_ = true;
  }
  allocator_ = device_->GetAllocator(alloc_attrs_);
}

Status TensorResponse::InitFrom(RecvTensorResponse* response) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_4(mht_4_v, 234, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::InitFrom");

  Status s;
  meta_.Swap(response);
  if (on_host_) {
    if (!tensor_.FromProto(allocator_, meta_.tensor())) {
      s = errors::InvalidArgument("Cannot parse tensor from response");
    }
  } else {
    s = device_->MakeTensorFromProto(meta_.tensor(), alloc_attrs_, &tensor_);
  }
  {
    TensorProto empty;
    meta_.mutable_tensor()->Swap(&empty);
  }
  meta_.clear_tensor();
  return s;
}

void TensorResponse::InitPartial(const RecvTensorResponse& response,
                                 const AllocationAttributes& allocation_attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::InitPartial");

  // Everything except content is present in *response.  Content will
  // arrive later; allocate a Tensor with appropriate storage for that
  // content.
  meta_ = response;
  TensorShape shape(meta_.tensor().tensor_shape());
  Tensor t(allocator_, meta_.tensor().dtype(), shape, allocation_attr);
  tensor_ = std::move(t);
}

Status TensorResponse::ParseFrom(Source* source) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_6(mht_6_v, 269, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::ParseFrom");

  if (!on_host_) {
    protobuf::io::CodedInputStream input(source->contents());

    // Pre-parse into local storage, then delegate to device.
    if (!meta_.ParseFromCodedStream(&input) || !input.ConsumedEntireMessage()) {
      return errors::InvalidArgument("Cannot parse tensor from response");
    }
    Status s =
        device_->MakeTensorFromProto(meta_.tensor(), alloc_attrs_, &tensor_);
    // Reduce memory usage for big tensors.
    {
      TensorProto empty;
      meta_.mutable_tensor()->Swap(&empty);
    }
    meta_.clear_tensor();
    return s;
  }
  if (already_used_) {
    ClearTensor();
  }
  already_used_ = true;
  if (ParseFast(source)) return Status::OK();
  meta_.Clear();
  if (ParseSlow(source)) return Status::OK();
  return errors::InvalidArgument("Cannot parse tensor from response");
}

// Define some helper routines for decoding protocol buffer wire format data
namespace {
// We only need some of the wiretype values for this code
enum WireType {
  WIRETYPE_VARINT = 0,
  WIRETYPE_LENGTH_DELIMITED = 2,
};
inline int GetTagFieldNumber(uint32 tag) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "GetTagFieldNumber");
 return tag >> 3; }
inline WireType GetTagWireType(uint32 tag) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_8(mht_8_v, 311, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "GetTagWireType");

  return static_cast<WireType>(tag & 0x7);
}

bool ReadVarintSizeAsInt(protobuf::io::CodedInputStream* input, int* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_9(mht_9_v, 318, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "ReadVarintSizeAsInt");

  protobuf_uint64 v;
  if (input->ReadVarint64(&v) && v <= static_cast<uint64>(INT_MAX)) {
    *result = static_cast<int>(v);
    return true;
  } else {
    return false;
  }
}

bool ReadNestedMessage(protobuf::io::CodedInputStream* input,
                       protobuf::Message* value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_10(mht_10_v, 332, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "ReadNestedMessage");

  int length;
  if (!ReadVarintSizeAsInt(input, &length)) return false;
  std::pair<protobuf::io::CodedInputStream::Limit, int> p =
      input->IncrementRecursionDepthAndPushLimit(length);
  if (p.second < 0 || !value->MergePartialFromCodedStream(input)) return false;
  // Make sure that parsing stopped when the limit was hit, not at an endgroup
  // tag.
  return input->DecrementRecursionDepthAndPopLimit(p.first);
}

}  // namespace

bool TensorResponse::ParseTensorSubmessage(
    protobuf::io::CodedInputStream* input, TensorProto* tensor_meta) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_11(mht_11_v, 349, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::ParseTensorSubmessage");

  bool seen_tensor_content = false;
  while (true) {
    auto p = input->ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      bool ok = (tag == 0);
      if (ok && !seen_tensor_content) {
        // No tensor content: could be because it's a zero-length tensor
        TensorShape shape(tensor_meta->tensor_shape());
        Tensor t(allocator_, tensor_meta->dtype(), shape);
        tensor_ = std::move(t);
      }
      return ok;
    }
    switch (tag) {
      case TensorProto::kDtypeFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input->ReadVarint32(&v)) return false;
        if (seen_tensor_content) return false;
        tensor_meta->set_dtype(static_cast<DataType>(static_cast<int>(v)));
        if (!DataTypeCanUseMemcpy(tensor_meta->dtype())) return false;
        break;
      }
      case TensorProto::kTensorShapeFieldNumber: {
        if ((wt != WIRETYPE_LENGTH_DELIMITED) ||
            !ReadNestedMessage(input, tensor_meta->mutable_tensor_shape()))
          return false;
        if (seen_tensor_content) return false;
        break;
      }
      case TensorProto::kVersionNumberFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input->ReadVarint32(&v)) return false;
        if (seen_tensor_content) return false;
        tensor_meta->set_version_number(static_cast<int32>(v));
        break;
      }
      case TensorProto::kTensorContentFieldNumber: {
        // If we haven't seen the dtype and tensor_shape data first, we can't
        // deal with this in the fast path.
        if (seen_tensor_content) return false;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !tensor_meta->has_tensor_shape()) {
          return false;
        }
        int num_bytes;
        if (!ReadVarintSizeAsInt(input, &num_bytes)) return false;
        seen_tensor_content = true;
        TensorShape shape(tensor_meta->tensor_shape());
        Tensor t(allocator_, tensor_meta->dtype(), shape);
        StringPiece buf = t.tensor_data();
        if (static_cast<size_t>(num_bytes) != buf.size()) return false;
        // TODO(jeff,sanjay): Figure out a way to avoid this copy if
        // the underlying ZeroCopyInputStream data is properly aligned
        // and compatible with what allocator_ wants.
        if (!input->ReadRaw(const_cast<char*>(buf.data()), num_bytes))
          return false;
        tensor_ = std::move(t);
        break;
      }
      default: {
        // Some other tag our fast path code is not prepared to handle.
        // return false.
        return false;
      }
    }
  }
}

bool TensorResponse::ParseFast(Source* source) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_12(mht_12_v, 423, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::ParseFast");

  protobuf::io::CodedInputStream input(source->contents());
  while (true) {
    auto p = input.ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      return (tag == 0);
    }
    switch (tag) {
      case RecvTensorResponse::kTensorFieldNumber: {
        if (wt != WIRETYPE_LENGTH_DELIMITED) return false;

        int length;
        if (!ReadVarintSizeAsInt(&input, &length)) return false;
        std::pair<protobuf::io::CodedInputStream::Limit, int> p =
            input.IncrementRecursionDepthAndPushLimit(length);
        if (p.second < 0 ||
            !ParseTensorSubmessage(&input, meta_.mutable_tensor())) {
          return false;
        }
        if (!input.DecrementRecursionDepthAndPopLimit(p.first)) {
          return false;
        }
        break;
      }
      case RecvTensorResponse::kIsDeadFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) return false;
        meta_.set_is_dead(v != 0);
        break;
      }
      case RecvTensorResponse::kSendStartMicrosFieldNumber: {
        protobuf_uint64 v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) return false;
        meta_.set_send_start_micros(static_cast<int64_t>(v));
        break;
      }
      case RecvTensorResponse::kTransportOptionsFieldNumber: {
        if ((wt != WIRETYPE_LENGTH_DELIMITED) ||
            !ReadNestedMessage(&input, meta_.mutable_transport_options()))
          return false;
        break;
      }
      case RecvTensorResponse::kRequireAckFieldNumber: {
        uint32 v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) return false;
        meta_.set_require_ack(v != 0);
        break;
      }
      default: {
        // Unknown tag, so don't handle we can't handle on the fast path
        return false;
      }
    }
  }

  return false;
}

bool TensorResponse::ParseSlow(Source* source) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePStensor_codingDTcc mht_13(mht_13_v, 486, "", "./tensorflow/core/distributed_runtime/tensor_coding.cc", "TensorResponse::ParseSlow");

  if (!meta_.ParseFromZeroCopyStream(source->contents())) {
    return false;
  }

  Tensor parsed(meta_.tensor().dtype());
  if (!parsed.FromProto(allocator_, meta_.tensor())) {
    return false;
  }
  tensor_ = std::move(parsed);

  // Reduce memory usage for big tensors.
  {
    TensorProto empty;
    meta_.mutable_tensor()->Swap(&empty);
  }
  meta_.clear_tensor();

  return true;
}

}  // namespace tensorflow
