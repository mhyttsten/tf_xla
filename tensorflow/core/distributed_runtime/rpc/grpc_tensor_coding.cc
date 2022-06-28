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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"

#include "grpcpp/support/byte_buffer.h"
#include "grpcpp/support/slice.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/proto_encode_helper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {
namespace grpc {

void EncodeRecvTensorResponseToByteBuffer(const RecvTensorResponse& proto,
                                          ::grpc::ByteBuffer* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.cc", "EncodeRecvTensorResponseToByteBuffer");

  ::grpc::Slice slice(proto.ByteSizeLong());
  proto.SerializeWithCachedSizesToArray(
      const_cast<uint8*>(reinterpret_cast<const uint8*>(slice.begin())));
  ::grpc::ByteBuffer tmp(&slice, 1);
  result->Swap(&tmp);
}

// We generate a RecvTensorResponse protocol buffer encoding into "*result",
// but where possible, we share the underlying Tensor buffer for "val", to
// avoid an extra copy.
//
// We hand-encode the protocol buffer data in the following order, as follows:
//
// Let R be a RecvTensorResponse object we want to encode, logically
// constructed by filling in data from "is_dead" and "val" and filling
// in a few other fields as well.
//
// (Letters here are used in the code to refer back to which part of the
//  encoding the code is generating).
//
// A:   <protocol buffer encoding of fields except R.tensor()>
// B1:  <tag encoding for RecvTensorResponse::tensor>
// B2:  <varint32 length of R.tensor() sub message>
// C:   <protocol buffer encoding of R.tensor() except for
//          R.tensor().tensor_content()>
// D1:  <tag encoding for TensorProto::tensor_content>
// D2:  <varint32 length of R.tensor().tensor_content() data>
// E:   <actual data for val's representation>
//
// If the tensor data is up to "kLargeTensorBytes", then A
// through E will all be encoded into "*result" in a single grpc::Slice.
//
// If the tensor data is larger than "kLargeTensorBytes", then A through
// D2 will be encoded in one grpc::Slice, and E will be encoded in a second
// grpc::Slice that points to the backing store for the tensor data, to avoid
// copying the tensor data (and the grpc::Slice setup will be arrange so as
// to dereference the underlying tensor data buffer when it is no longer
// needed in the "*result" ByteBuffer).
static int VarLengthEncodingSize(uint32 tag, size_t bytes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.cc", "VarLengthEncodingSize");

  return core::VarintLength(tag << 3) + core::VarintLength(bytes) + bytes;
}

// Returns an upper bound in bytes of the protocol buffer encoding of
// the "skeleton" of "val" (all the data needed for dtype and the shape,
// but not the actual contents of "val").
static int SkeletonEncodingSizeUpperBound(const Tensor& val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.cc", "SkeletonEncodingSizeUpperBound");

  static const int kVarintMax64 = 10;  // Max length of varint64 encoding
  const int ndims = val.shape().dims();
  return (2 * kVarintMax64) +           // dtype
         (ndims * (4 * kVarintMax64));  // Shape: 4 varints per dim
}

// Encode the skeleton for "val" (the encoded TensorProto contents
// (dtype and shape, but not the actual data) into "*e".  The backing
// store for "*e" must be of appropriate size to hold this encoding.
static void EncodeSkeleton(const Tensor& val, io::ProtoEncodeHelper* e) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.cc", "EncodeSkeleton");

  // Encode val.dtype()
  e->WriteUint64(TensorProto::kDtypeFieldNumber, val.dtype());

  // Compute length of val.shape() proto encoding
  const int ndims = val.shape().dims();
  int tensor_shape_bytes = 0;
  for (int d = 0; d < ndims; d++) {
    int64_t dim_size = val.shape().dim_size(d);
    tensor_shape_bytes +=
        2 +  // TensorShapeProto dim tag + varintlength of submessage
        1 +  // TensorShapeProto_Dim::kSizeFieldNumber
        core::VarintLength(dim_size);
  }

  if (tensor_shape_bytes > 0) {
    e->WriteVarlengthBeginning(TensorProto::kTensorShapeFieldNumber,
                               tensor_shape_bytes);
    // Encode val.shape()
    for (int d = 0; d < ndims; d++) {
      int64_t dim_size = val.shape().dim_size(d);
      int64_t dim_varlen = 1 +  // TensorShapeProto_Dim::kSizeFieldNumber
                           core::VarintLength(dim_size);
      e->WriteVarlengthBeginning(TensorShapeProto::kDimFieldNumber, dim_varlen);
      e->WriteUint64(TensorShapeProto_Dim::kSizeFieldNumber, dim_size);
    }
  }

#ifndef NDEBUG
  {
    // Debug-mode only check to make sure the encoding above is
    // identical to the auto-generated protocol buffer encoding.
    TensorProto skeleton;
    skeleton.set_dtype(val.dtype());
    val.shape().AsProto(skeleton.mutable_tensor_shape());
    string tensor_except_contents;  // tensor() field except contents
    skeleton.AppendToString(&tensor_except_contents);
    TensorProto skeleton2;
    skeleton2.ParseFromString(string(e->data(), e->size()));
    string out;
    skeleton.AppendToString(&out);
    DCHECK_EQ(tensor_except_contents, out) << skeleton.DebugString() << " vs\n"
                                           << skeleton2.DebugString();
  }
#endif
}

void EncodeTensorToByteBuffer(bool is_dead, const Tensor& val, bool require_ack,
                              ::grpc::ByteBuffer* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_tensor_codingDTcc mht_4(mht_4_v, 319, "", "./tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.cc", "EncodeTensorToByteBuffer");

  const int kLargeTensorBytes = 1024;
  const int64_t kProtoBufLimitBytes = 1LL << 31;

  if (val.TotalBytes() > kProtoBufLimitBytes) {
    size_t exceeded_bytes = val.TotalBytes() - kProtoBufLimitBytes;
    LOG(FATAL) << "Cannot encode a Tensor that exceeds the 2GB protobuf limit. "
                  "Exceeded bytes: "
               << exceeded_bytes;
  }

  RecvTensorResponse response;
  if (is_dead) {
    response.set_is_dead(is_dead);
  }
  response.set_require_ack(require_ack);
  response.set_send_start_micros(Env::Default()->NowMicros());
  if (!DataTypeCanUseMemcpy(val.dtype())) {
    // Straightforward but slow path for complicated kinds of tensor data
    // TODO(jeff,sanjay): If this becomes an issue, we could
    // go directly from val -> ByteBuffer, with some effort.
    val.AsProtoTensorContent(response.mutable_tensor());

    // Encode full protocol buffer to a ByteBuffer
    EncodeRecvTensorResponseToByteBuffer(response, result);
  } else {
    // skeleton is the encoded TensorProto contents (dtype and shape), but
    // not the actual data
    gtl::InlinedVector<char, 128> skeleton(SkeletonEncodingSizeUpperBound(val));
    io::ProtoEncodeHelper e_skeleton(skeleton.data(), skeleton.size());
    EncodeSkeleton(val, &e_skeleton);

    StringPiece tdata = val.tensor_data();
    uint32 overall_tensor_proto_bytesize =
        (e_skeleton.size() +
         VarLengthEncodingSize(TensorProto::kTensorContentFieldNumber,
                               tdata.size()));
    string header;  // All of RecvTensorResponse except the tensor() field
    response.AppendToString(&header);

    size_t expected_size =
        (header.size() +
         VarLengthEncodingSize(RecvTensorResponse::kTensorFieldNumber,
                               overall_tensor_proto_bytesize));
    // If "share_tensor_slice_memory == false", we copy the tensor data to
    // the end of the buffer we are preparing that holds the rest of the
    // RecvTensorResponse protocol buffer.
    //
    // If "share_tensor_slice_memory == true", we arrange to share the
    // backing store of the data by creating a slice that also points to the
    // backing store, with appropriate reference counts to keep the
    // backing store alive as needed.
    //
    // We enable this behavior if the tensor is large.
    bool share_tensor_slice_memory = (tdata.size() > kLargeTensorBytes);

    size_t encoder_size = expected_size - tdata.size();

    // Encode all but the actual "tdata", but including the tag and
    // varlength header for the "tdata"
    gtl::InlinedVector<char, 1024> space(encoder_size);
    io::ProtoEncodeHelper e(space.data(), space.size());
    // (A)
    e.WriteRawBytes(header);

    // (B1) & (B2)
    e.WriteVarlengthBeginning(RecvTensorResponse::kTensorFieldNumber,
                              overall_tensor_proto_bytesize);
    // (C)
    e.WriteRawBytes(StringPiece(e_skeleton.data(), e_skeleton.size()));
    // (D1) & (D2)
    e.WriteVarlengthBeginning(TensorProto::kTensorContentFieldNumber,
                              tdata.size());

    // All but the tensor backing store are serialized now

    // Now allocate memory and put into the ByteBuffer
    ::grpc::Slice slices[2];
    int num_slices = 0;
    {
      size_t slice_len =
          e.size() + (share_tensor_slice_memory ? 0 : tdata.size());
      slices[0] = ::grpc::Slice(slice_len);
      memcpy(const_cast<uint8_t*>(slices[0].begin()), e.data(), e.size());
      if (!share_tensor_slice_memory) {
        // (E)
        memcpy(const_cast<uint8_t*>(slices[0].begin()) + e.size(), tdata.data(),
               tdata.size());
      }
      num_slices += 1;
    }

    if (share_tensor_slice_memory) {
      // (E) Encode tensor data, but by sharing backing store
      const TensorBuffer* buf = DMAHelper::buffer(&val);
      buf->Ref();
      slices[1] = ::grpc::Slice(
          const_cast<void*>(static_cast<const void*>(tdata.data())),
          tdata.size(),
          [](void* backing) { static_cast<TensorBuffer*>(backing)->Unref(); },
          const_cast<TensorBuffer*>(buf));
      num_slices += 1;
    }
    size_t total_bytes = 0;
    for (int i = 0; i < num_slices; i++) {
      total_bytes += slices[i].size();
    }
    CHECK_EQ(total_bytes, expected_size);

    ::grpc::ByteBuffer tmp(&slices[0], num_slices);
    result->Swap(&tmp);
  }
}

}  // namespace grpc
}  // namespace tensorflow
