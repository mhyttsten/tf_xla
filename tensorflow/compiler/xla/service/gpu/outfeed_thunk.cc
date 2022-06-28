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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSoutfeed_thunkDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSoutfeed_thunkDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSoutfeed_thunkDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

OutfeedThunk::OutfeedThunk(ThunkInfo thunk_info,
                           std::vector<ShapedSlice> source_slices)
    : Thunk(Kind::kOutfeed, thunk_info),
      source_slices_(std::move(source_slices)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSoutfeed_thunkDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/xla/service/gpu/outfeed_thunk.cc", "OutfeedThunk::OutfeedThunk");
}

Status OutfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSoutfeed_thunkDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/xla/service/gpu/outfeed_thunk.cc", "OutfeedThunk::ExecuteOnStream");

  se::Stream& stream = *params.stream;
  const BufferAllocations& buffer_allocations = *params.buffer_allocations;

  VLOG(2) << "Outfeeding from GPU";

  OutfeedManager* outfeed_manager = GetOrCreateOutfeedManager(stream.parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* output_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Nothing to be done for an outfeed with no inputs.
  // Note: Cannot do this before `BlockingGetNextDestination` above to dequeue
  // an entry from the outfeed manager.
  if (source_slices_.empty()) {
    return Status::OK();
  }

  const int64_t leaf_count = output_buffers->leaf_count();
  TF_RET_CHECK(source_slices_.size() == leaf_count)
      << "Mismatch between number of outfeed inputs (" << source_slices_.size()
      << ") and outputs (" << leaf_count << ")";

  auto output_leaf_it = output_buffers->leaf_begin();
  for (int64_t index = 0; index < leaf_count; ++index) {
    // Assert that the shapes are compatible.
    const ShapeIndex& shape_index = output_leaf_it->first;
    std::unique_ptr<OutfeedBuffer>& buffer = output_leaf_it->second;

    // NOTE: This code needs deal with the `output_buffers` object getting
    // deleted when its executing. Specifically, objects in the outfeed queue
    // are pointers to instance of stack allocated objects in
    // `GpuTransferManager::TransferLiteralFromOutfeed`. When all leaf node
    // buffers are notified via "buffer->Done()" below in the stream host
    // callback, `TransferLiteralFromOutfeed` deletes this stack allocated
    // object when it returns. This means that its possible that during the last
    // iteration, after the call to "buffer->Done()" is scheduled onto the
    // stream, the `output_buffers` object might get deleted, so we should avoid
    // accessing the object after that.
    //
    // To achieve that, increment the leaf iterator here before the last "Done"
    // is enqueued, instead of in the loop increment, which would be after the
    // "Done" is scheduled.
    ++output_leaf_it;
    const Shape& output_shape =
        ShapeUtil::GetSubshape(output_buffers->shape(), shape_index);
    TF_RET_CHECK(ShapeUtil::Equal(source_slices_[index].shape, output_shape))
        << "Mismatch between outfeed output buffer shape "
        << ShapeUtil::HumanStringWithLayout(output_shape)
        << " and outfeed source buffer shape "
        << ShapeUtil::HumanStringWithLayout(source_slices_[index].shape);

    BufferAllocation::Slice source_slice = source_slices_[index].slice;
    if (!source_slice.allocation())
      return InternalError("outfeed source missing buffer allocation");
    se::DeviceMemoryBase data_address =
        buffer_allocations.GetDeviceAddress(source_slice);

    // TODO(b/111309141): Run this on a separate stream so it doesn't block
    // the GPU from doing work during the transfer. This could be handled by
    // making StreamAssignment do something intelligent with outfeed thunks.
    stream
        .ThenMemcpy(buffer->destination()->untyped_data(), data_address,
                    buffer->length())
        .ThenDoHostCallback([&buffer]() { buffer->Done(); });
  }

  Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         &stream, block_status.error_message());
  }

  VLOG(2) << "Outfeeding from GPU complete";
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
