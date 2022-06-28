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
class MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using absl::StrFormat;
using ::xla::BufferAllocationProto;
using ::xla::HloInstructionProto;
using ::xla::HloProto;
using ::xla::LayoutUtil;
using ::xla::LogicalBufferProto;
using ::xla::Shape;
using ::xla::ShapeUtil;

double BytesToMiB(int64_t bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "BytesToMiB");

  return static_cast<double>(bytes) / tensorflow::MathUtil::IPow(2, 20);
}

// Get buffer allocation property.
std::string GetAllocationGroupName(
    const BufferAllocationProto* buffer_allocation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "GetAllocationGroupName");

  if (buffer_allocation == nullptr) {
    return "";
  }
  if (buffer_allocation->is_entry_computation_parameter()) {
    return "Parameter";
  } else if (buffer_allocation->maybe_live_out()) {
    return "Output";
  } else if (buffer_allocation->is_thread_local()) {
    return "Thread-local";
  } else {
    return "Temporary";
  }
}

// Get the instruction associated with the logical buffer.
std::string GetInstructionName(const LogicalBufferProto* logical_buffer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "GetInstructionName");

  if (logical_buffer == nullptr) {
    return "";
  }
  if (logical_buffer->defined_at().shape_index().empty()) {
    return logical_buffer->defined_at().instruction_name();
  } else {
    return absl::StrCat(
        logical_buffer->defined_at().instruction_name(), "{",
        absl::StrJoin(logical_buffer->defined_at().shape_index(), ""), "}");
  }
}

HeapObject MakeHeapObjectCommon(std::string label, int logical_buffer_id,
                                int64_t logical_buffer_size_bytes,
                                int64_t unpadded_shape_bytes) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "MakeHeapObjectCommon");

  HeapObject result;
  result.set_label(std::move(label));
  result.set_logical_buffer_id(logical_buffer_id);
  result.set_logical_buffer_size_mib(BytesToMiB(logical_buffer_size_bytes));
  result.set_unpadded_shape_mib(BytesToMiB(unpadded_shape_bytes));
  return result;
}

HeapObject MakeHeapObject(const std::string& tf_op_name,
                          const std::string& shape_string,
                          const std::string& op_code,
                          std::string instruction_name, std::string group_name,
                          std::string label, int color, int logical_buffer_id,
                          int64_t logical_buffer_size_bytes,
                          int64_t unpadded_shape_bytes) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("tf_op_name: \"" + tf_op_name + "\"");
   mht_4_v.push_back("shape_string: \"" + shape_string + "\"");
   mht_4_v.push_back("op_code: \"" + op_code + "\"");
   mht_4_v.push_back("instruction_name: \"" + instruction_name + "\"");
   mht_4_v.push_back("group_name: \"" + group_name + "\"");
   mht_4_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "MakeHeapObject");

  HeapObject result =
      MakeHeapObjectCommon(std::move(label), logical_buffer_id,
                           logical_buffer_size_bytes, unpadded_shape_bytes);
  result.set_numbered(color);
  result.set_instruction_name(std::move(instruction_name));
  result.set_group_name(std::move(group_name));
  result.set_tf_op_name(tf_op_name);
  result.set_shape_string(shape_string);
  result.set_op_code(op_code);
  return result;
}

HeapObject MakeHeapObject(std::string color, std::string label,
                          int logical_buffer_id,
                          int64_t logical_buffer_size_bytes,
                          int64_t unpadded_shape_bytes) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("color: \"" + color + "\"");
   mht_5_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_5(mht_5_v, 315, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "MakeHeapObject");

  HeapObject result =
      MakeHeapObjectCommon(std::move(label), logical_buffer_id,
                           logical_buffer_size_bytes, unpadded_shape_bytes);
  result.set_named(std::move(color));
  return result;
}

BufferSpan MakeBufferSpan(int32 start, int32 limit) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_6(mht_6_v, 326, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "MakeBufferSpan");

  BufferSpan result;
  result.set_start(start);
  result.set_limit(limit);
  return result;
}

const Shape* ResolveShapeIndex(const Shape* shape,
                               absl::Span<const int64_t> shape_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_7(mht_7_v, 337, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "ResolveShapeIndex");

  for (int64_t value : shape_index) {
    shape = &shape->tuple_shapes(value);
  }
  return shape;
}

// A wrapper around ShapeUtil::ByteSizeOf that clears out the layout/padding,
// since that is considered in the ByteSizeOf calculation.
int64_t UnpaddedSize(Shape shape) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_8(mht_8_v, 349, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "UnpaddedSize");

  // Ensure the layout has no padding by making it the default layout.
  LayoutUtil::SetToDefaultLayout(&shape);
  // Note: we make a simplifying assumption here that a "minimal" size for a
  // tuple member would be the size of a `void*` -- there may be even fancier
  // ways of doing things, but this should give a good enough approximation of
  // what a minimal tuple size is.
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/sizeof(void*));
}

void Convert(const xla::BufferAllocationProto_Assigned& assigned,
             const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
                 id_to_logical_buffer,
             const absl::node_hash_map<std::string, const HloInstructionProto*>&
                 name_to_hlo,
             LogicalBuffer* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_9(mht_9_v, 367, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "Convert");

  result->set_id(assigned.logical_buffer_id()),
      result->set_size_mib(BytesToMiB(assigned.size()));
  const LogicalBufferProto* proto =
      id_to_logical_buffer.at(assigned.logical_buffer_id());
  const std::string& instruction_name = proto->defined_at().instruction_name();
  result->set_hlo_name(instruction_name);
  result->mutable_shape_index()->CopyFrom(proto->defined_at().shape_index());
  const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
  const Shape* shape =
      ResolveShapeIndex(&top_level_shape, proto->defined_at().shape_index());
  result->set_shape(ShapeUtil::HumanStringWithLayout(*shape));
}

bool IsReusable(const BufferAllocationProto& buffer_allocation) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_10(mht_10_v, 384, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "IsReusable");

  return !buffer_allocation.is_thread_local() && !buffer_allocation.is_tuple();
}

void Convert(const BufferAllocationProto& proto,
             const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
                 id_to_logical_buffer,
             const absl::node_hash_map<std::string, const HloInstructionProto*>&
                 name_to_hlo,
             BufferAllocation* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_11(mht_11_v, 396, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "Convert");

  result->set_id(proto.index());
  result->set_size_mib(BytesToMiB(proto.size()));
  if (proto.is_entry_computation_parameter()) {
    result->add_attributes("entry computation parameter");
  }
  if (proto.maybe_live_out()) {
    result->add_attributes("may-be live out");
  }
  if (IsReusable(proto)) {
    result->add_attributes("reusable");
  }
  for (const auto& assigned : proto.assigned()) {
    Convert(assigned, id_to_logical_buffer, name_to_hlo,
            result->add_logical_buffers());
  }
  // Check whether all logical buffers for this buffer allocation have a common
  // shape.
  if (!result->logical_buffers().empty()) {
    std::string common_shape = result->logical_buffers(0).shape();
    for (int64_t i = 1; i < result->logical_buffers_size(); ++i) {
      if (result->logical_buffers(i).shape() != common_shape) {
        common_shape = "";
        break;
      }
    }
    if (!common_shape.empty()) {
      result->set_common_shape(common_shape);
    }
  }
}

void NoteSpecialAllocations(
    const absl::flat_hash_set<const BufferAllocationProto*>&
        all_buffer_allocations,
    const absl::flat_hash_map<int64_t, const LogicalBufferProto*>&
        id_to_logical_buffer,

    const absl::node_hash_map<std::string, const HloInstructionProto*>&
        name_to_hlo,
    int64_t small_buffer_size, PreprocessResult* result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_12(mht_12_v, 439, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "NoteSpecialAllocations");

  int64_t entry_parameters_bytes = 0;
  int64_t non_reusable_bytes = 0;
  int64_t maybe_live_out_bytes = 0;
  for (const BufferAllocationProto* buffer_allocation :
       all_buffer_allocations) {
    if (buffer_allocation->is_entry_computation_parameter()) {
      entry_parameters_bytes += buffer_allocation->size();
    }
    if (!IsReusable(*buffer_allocation)) {
      non_reusable_bytes += buffer_allocation->size();
    }
    if (buffer_allocation->maybe_live_out()) {
      if (buffer_allocation->size() > small_buffer_size) {
        VLOG(1) << "Maybe live out buffer allocation: "
                << buffer_allocation->size()
                << " bytes :: " << buffer_allocation->ShortDebugString();
      }
      maybe_live_out_bytes += buffer_allocation->size();
    }
    Convert(*buffer_allocation, id_to_logical_buffer, name_to_hlo,
            result->add_indefinite_lifetimes());
  }

  result->set_entry_computation_parameters_mib(
      BytesToMiB(entry_parameters_bytes));
  result->set_non_reusable_mib(BytesToMiB(non_reusable_bytes));
  result->set_maybe_live_out_mib(BytesToMiB(maybe_live_out_bytes));
}

}  // namespace

absl::StatusOr<PreprocessResult> ConvertHloProtoToPreprocessResult(
    const HloProto& hlo_proto, int64_t small_buffer_size,
    int64_t heap_simulator_trace_id, int64_t memory_color) {
  // Construct a mapping from name to HLO proto.
  absl::node_hash_map<std::string, const HloInstructionProto*> name_to_hlo;
  for (const auto& computation : hlo_proto.hlo_module().computations()) {
    for (const auto& instruction : computation.instructions()) {
      name_to_hlo[instruction.name()] = &instruction;
      VLOG(1) << "HLO: " << instruction.ShortDebugString();
    }
  }

  // Mapping from logical buffer ID to logical buffer, and set of all logical
  // buffer protos.
  absl::flat_hash_map<int64_t, const LogicalBufferProto*> id_to_logical_buffer;
  absl::flat_hash_set<const LogicalBufferProto*> all_logical_buffers;
  for (const auto& logical_buffer :
       hlo_proto.buffer_assignment().logical_buffers()) {
    VLOG(1) << "Logical buffer: " << logical_buffer.ShortDebugString();
    id_to_logical_buffer[logical_buffer.id()] = &logical_buffer;
    all_logical_buffers.insert(&logical_buffer);
  }

  // Mapping from logocal buffer proto to the buffer allocation that it exists
  // inside (there must be only one).
  //
  // Also a reverse mapping from buffer allocation proto to the set of logical
  // buffer protos that exist inside of it.
  absl::flat_hash_map<const LogicalBufferProto*, const BufferAllocationProto*>
      logical_buffer_to_buffer_allocation;
  absl::node_hash_map<const BufferAllocationProto*,
                      absl::flat_hash_set<const LogicalBufferProto*>>
      buffer_allocation_to_logical_buffers;
  absl::flat_hash_set<const BufferAllocationProto*> all_buffer_allocations;
  for (const BufferAllocationProto& buffer_allocation :
       hlo_proto.buffer_assignment().buffer_allocations()) {
    all_buffer_allocations.insert(&buffer_allocation);
    for (const xla::BufferAllocationProto_Assigned& assigned :
         buffer_allocation.assigned()) {
      const LogicalBufferProto* logical_buffer =
          id_to_logical_buffer.at(assigned.logical_buffer_id());
      buffer_allocation_to_logical_buffers[&buffer_allocation].insert(
          logical_buffer);
      auto insert_result = logical_buffer_to_buffer_allocation.insert(
          {logical_buffer, &buffer_allocation});
      if (!insert_result.second) {
        return absl::InvalidArgumentError(
            "A logical buffer appears to be associated with multiple buffer "
            "allocations.");
      }
    }
  }

  std::vector<int64_t> logical_buffers;
  std::vector<int64_t> peak_logical_buffers;

  int64_t heap_size_bytes = 0;
  int64_t unpadded_heap_size_bytes = 0;

  int64_t peak_heap_size_bytes = 0;
  int64_t unpadded_peak_heap_size_bytes = 0;  // Unpadded size at peak.
  int64_t peak_heap_size_position = 0;
  std::vector<double> heap_sizes;
  std::vector<double> unpadded_heap_sizes;

  absl::node_hash_map<int64_t, std::pair<int64_t, absl::optional<int64_t>>>
      logical_buffer_spans;
  absl::flat_hash_set<const LogicalBufferProto*> seen;
  absl::flat_hash_set<const BufferAllocationProto*> seen_buffer_allocations;

  // Run through all the simulator events in the given trace, and simulate the
  // heap in order to find the point of peak memory usage and record its
  // associated metadata.
  if (heap_simulator_trace_id >= 0 &&
      heap_simulator_trace_id <
          hlo_proto.buffer_assignment().heap_simulator_traces_size()) {
    const auto& simulator_events =
        hlo_proto.buffer_assignment()
            .heap_simulator_traces(heap_simulator_trace_id)
            .events();
    for (const auto& event : simulator_events) {
      heap_sizes.push_back(BytesToMiB(heap_size_bytes));
      unpadded_heap_sizes.push_back(BytesToMiB(unpadded_heap_size_bytes));
      const auto* logical_buffer = id_to_logical_buffer.at(event.buffer_id());
      seen.insert(logical_buffer);
      seen_buffer_allocations.insert(
          logical_buffer_to_buffer_allocation.at(logical_buffer));
      const auto& instruction_name =
          logical_buffer->defined_at().instruction_name();
      const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
      const Shape* shape = ResolveShapeIndex(
          &top_level_shape, logical_buffer->defined_at().shape_index());
      if (event.kind() == xla::HeapSimulatorTrace_Event::ALLOC ||
          event.kind() == xla::HeapSimulatorTrace_Event::SHARE_WITH) {
        logical_buffers.push_back(event.buffer_id());
        heap_size_bytes += logical_buffer->size();
        unpadded_heap_size_bytes += UnpaddedSize(*shape);
        // Initialize the buffer span from the current event to the last event.
        logical_buffer_spans[event.buffer_id()] = {heap_sizes.size() - 1,
                                                   simulator_events.size() - 1};
        int64_t prior_peak_heap_size_bytes = peak_heap_size_bytes;
        peak_heap_size_bytes = std::max(peak_heap_size_bytes, heap_size_bytes);
        if (prior_peak_heap_size_bytes != peak_heap_size_bytes) {
          peak_heap_size_position = heap_sizes.size() - 1;
          unpadded_peak_heap_size_bytes = unpadded_heap_size_bytes;
          VLOG(1) << StrFormat("New peak heap size on %d: %s :: %d bytes",
                               peak_heap_size_position, instruction_name,
                               peak_heap_size_bytes);
          peak_logical_buffers = logical_buffers;
        }
      } else if (event.kind() == xla::HeapSimulatorTrace_Event::FREE) {
        logical_buffers.erase(
            std::remove(logical_buffers.begin(), logical_buffers.end(),
                        event.buffer_id()),
            logical_buffers.end());
        heap_size_bytes -= logical_buffer->size();
        unpadded_heap_size_bytes -= UnpaddedSize(*shape);
        logical_buffer_spans[event.buffer_id()].second = heap_sizes.size() - 1;
        if (heap_size_bytes < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "heap_size_bytes should be non-negative: ", heap_size_bytes));
        }
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unhandled event kind: ", event.kind()));
      }
    }

    // Add the final heap size after simulating the entire heap trace.
    heap_sizes.push_back(BytesToMiB(heap_size_bytes));
    unpadded_heap_sizes.push_back(BytesToMiB(unpadded_heap_size_bytes));

    if (seen_buffer_allocations.size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("All heap simulation should work out of a single buffer "
                       "allocation, actual seen_buffer_allocations.size():",
                       seen_buffer_allocations.size()));
    }
  }

  VLOG(1) << "Found " << peak_logical_buffers.size()
          << " logical buffers alive at point of peak heap usage.";

  VLOG(1) << "Peak logical buffers: ["
          << absl::StrJoin(peak_logical_buffers, ",") << "]";

  int64_t indefinite_memory_usage_bytes = 0;
  std::vector<HeapObject> max_heap;
  int colorno = 0;
  int64_t rest = 0;

  // Helper lambda that adds the logical buffer as an element in the "max heap"
  // view with constitutent logical buffers.
  auto add_heap_object = [&](const LogicalBufferProto* logical_buffer,
                             const BufferAllocationProto* buffer_allocation) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_13(mht_13_v, 628, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "lambda");

    if (logical_buffer->size() <= small_buffer_size) {
      rest += logical_buffer->size();
      return;
    }
    const std::string& instruction_name =
        logical_buffer->defined_at().instruction_name();
    const Shape top_level_shape(name_to_hlo.at(instruction_name)->shape());
    const Shape* shape = ResolveShapeIndex(
        &top_level_shape, logical_buffer->defined_at().shape_index());
    std::string shape_string = ShapeUtil::HumanStringWithLayout(*shape);
    int64 unpadded_shape_bytes = UnpaddedSize(*shape);
    const HloInstructionProto* hlo_instruction =
        name_to_hlo.at(instruction_name);
    std::string label = StrFormat("%s: %s # %s", instruction_name, shape_string,
                                  hlo_instruction->metadata().op_name());
    max_heap.push_back(MakeHeapObject(
        hlo_instruction->metadata().op_name(), shape_string,
        hlo_instruction->opcode(), GetInstructionName(logical_buffer),
        GetAllocationGroupName(buffer_allocation), std::move(label), colorno++,
        logical_buffer->id(), logical_buffer->size(), unpadded_shape_bytes));
  };

  // Now look for all logical buffers which have not been seen, and assume they
  // have indefinite lifetime if they are not in thread-local buffer
  // allocations.
  absl::flat_hash_set<const LogicalBufferProto*> unseen;
  for (const LogicalBufferProto* logical_buffer : all_logical_buffers) {
    if (!seen.contains(logical_buffer)) {
      unseen.insert(logical_buffer);
    }
  }
  for (const LogicalBufferProto* logical_buffer : unseen) {
    const BufferAllocationProto* buffer_allocation =
        logical_buffer_to_buffer_allocation.at(logical_buffer);
    if (buffer_allocation->is_thread_local()) {
      continue;
    }
    if (logical_buffer->color() != memory_color) {
      continue;
    }
    // Clear out the assigned logical buffers when stringifying the buffer
    // allocation, as it can be a long list.
    auto to_string = [](const BufferAllocationProto* p) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_14(mht_14_v, 674, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "lambda");

      BufferAllocationProto copy = *p;
      copy.mutable_assigned()->Clear();
      return copy.ShortDebugString();
    };
    if (seen_buffer_allocations.insert(buffer_allocation).second) {
      indefinite_memory_usage_bytes += buffer_allocation->size();
      const auto& logical_buffers =
          buffer_allocation_to_logical_buffers.at(buffer_allocation);
      if (logical_buffers.size() == 1) {
        add_heap_object(*logical_buffers.begin(), buffer_allocation);
      } else {
        VLOG(1) << "Indefinite lifetime, no heap object shown due to "
                   "multiple logical buffers in buffer allocation: "
                << logical_buffer->ShortDebugString()
                << " :: " << to_string(buffer_allocation) << std::endl;
      }
      if (buffer_allocation->size() > small_buffer_size) {
        VLOG(1) << "Indefinite memory usage now: "
                << indefinite_memory_usage_bytes << " bytes (+"
                << buffer_allocation->size() << " bytes)";
      }
    }
  }

  // For the buffers that have indefinite lifetime (that is, lifetime not
  // reflected by the heap simulation) add it to the peak values and the vectors
  // of heap sizes.
  peak_heap_size_bytes += indefinite_memory_usage_bytes;
  unpadded_peak_heap_size_bytes += indefinite_memory_usage_bytes;
  double addend = BytesToMiB(indefinite_memory_usage_bytes);
  for (int i = 0; i < heap_sizes.size(); ++i) {
    heap_sizes[i] += addend;
    unpadded_heap_sizes[i] += addend;
  }

  // Accumulate data for use in a stacked bar plot.
  //
  // We accumulate it in "program order" -- the order in which it was placed
  // into the logical_buffers sequence above was program order, and we iterate
  // that order to create data points.
  for (int logical_buffer_id : peak_logical_buffers) {
    const auto* logical_buffer = id_to_logical_buffer.at(logical_buffer_id);
    const auto* buffer_allocation =
        logical_buffer_to_buffer_allocation.at(logical_buffer);
    add_heap_object(logical_buffer, buffer_allocation);
  }
  if (rest != 0) {
    max_heap.push_back(MakeHeapObject(
        "gray", StrFormat("small (<%d bytes)", small_buffer_size), -1, rest,
        0));
  }

  std::vector<const HeapObject*> max_heap_by_size;
  max_heap_by_size.reserve(max_heap.size());
  for (const auto& object : max_heap) {
    max_heap_by_size.push_back(&object);
  }
  std::sort(max_heap_by_size.begin(), max_heap_by_size.end(),
            [](const HeapObject* a, const HeapObject* b) {
              return a->logical_buffer_size_mib() >
                     b->logical_buffer_size_mib();
            });

  std::vector<int> max_heap_to_by_size;
  max_heap_to_by_size.reserve(max_heap.size());
  for (const auto& object : max_heap) {
    auto it =
        std::find(max_heap_by_size.begin(), max_heap_by_size.end(), &object);
    int index = std::distance(max_heap_by_size.begin(), it);
    max_heap_to_by_size.push_back(index);
  }

  std::vector<int> by_size_to_max_heap;
  for (const auto* object : max_heap_by_size) {
    int index = object - &max_heap[0];
    by_size_to_max_heap.push_back(index);
  }

  PreprocessResult result;
  result.set_module_name(hlo_proto.hlo_module().name());
  result.set_entry_computation_name(
      hlo_proto.hlo_module().entry_computation_name());
  *result.mutable_heap_sizes() = {heap_sizes.begin(), heap_sizes.end()};
  *result.mutable_unpadded_heap_sizes() = {unpadded_heap_sizes.begin(),
                                           unpadded_heap_sizes.end()};
  *result.mutable_max_heap() = {max_heap.begin(), max_heap.end()};
  for (const HeapObject* o : max_heap_by_size) {
    *result.add_max_heap_by_size() = *o;
  }
  *result.mutable_max_heap_to_by_size() = {max_heap_to_by_size.begin(),
                                           max_heap_to_by_size.end()};
  *result.mutable_by_size_to_max_heap() = {by_size_to_max_heap.begin(),
                                           by_size_to_max_heap.end()};
  result.set_peak_heap_mib(BytesToMiB(peak_heap_size_bytes));
  result.set_peak_unpadded_heap_mib(BytesToMiB(unpadded_peak_heap_size_bytes));
  result.set_peak_heap_size_position(peak_heap_size_position);

  for (const auto& item : logical_buffer_spans) {
    (*result.mutable_logical_buffer_spans())[item.first] =
        MakeBufferSpan(item.second.first, item.second.second.value());
  }

  NoteSpecialAllocations(all_buffer_allocations, id_to_logical_buffer,
                         name_to_hlo, small_buffer_size, &result);
  return result;
}

// From a list of heap simulator traces, identify the one that has the largest
// number of memory events with color <memory_color>.
// If unable to find the heap simulator trace, return -1, and
// ConvertHloProtoToPreprocessResult will not consider heap_simulator_traces
// during preprocess.
int64_t GetHeapSimulatorTraceIdFromEvents(const HloProto& proto,
                                          int64_t memory_color) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_15(mht_15_v, 791, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "GetHeapSimulatorTraceIdFromEvents");

  absl::flat_hash_map<int64_t, const xla::LogicalBufferProto*>
      id_to_logical_buffer;
  for (const auto& logical_buffer :
       proto.buffer_assignment().logical_buffers()) {
    id_to_logical_buffer[logical_buffer.id()] = &logical_buffer;
  }
  int64_t best_index = -1;
  int64_t best_event_count = 0;
  for (int64_t i = 0;
       i < proto.buffer_assignment().heap_simulator_traces_size(); i++) {
    const auto& heap_simulator_trace =
        proto.buffer_assignment().heap_simulator_traces(i);
    int64_t event_count = 0;
    for (const auto& event : heap_simulator_trace.events()) {
      const auto iter = id_to_logical_buffer.find(event.buffer_id());
      if (iter == id_to_logical_buffer.end()) {
        continue;
      }
      if (iter->second->color() == memory_color) {
        event_count++;
      }
    }
    if (event_count > best_event_count) {
      best_index = i;
      best_event_count = event_count;
    }
  }

  return best_index;
}

// Tries to get the correct heap simulator trace based on
// buffer_allocation_index.
int64_t GetHeapSimulatorTraceIdFromBufferAllocationIndex(const HloProto& proto,
                                                         int64_t memory_color) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_16(mht_16_v, 829, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "GetHeapSimulatorTraceIdFromBufferAllocationIndex");

  absl::flat_hash_map<int64_t, const xla::BufferAllocationProto*>
      id_to_buffer_allocation;
  for (const auto& buffer_allocation :
       proto.buffer_assignment().buffer_allocations()) {
    id_to_buffer_allocation[buffer_allocation.index()] = &buffer_allocation;
  }
  for (int64_t i = 0;
       i < proto.buffer_assignment().heap_simulator_traces_size(); ++i) {
    int64_t buffer_allocation_index = proto.buffer_assignment()
                                          .heap_simulator_traces(i)
                                          .buffer_allocation_index();
    const auto iter = id_to_buffer_allocation.find(buffer_allocation_index);
    if (buffer_allocation_index && iter != id_to_buffer_allocation.end()) {
      // Find the heap simulator trace that corresponds to the HLO temporaries
      // buffer allocation, where is_thread_local,
      // is_entry_computation_parameter, is_constant, and maybe_live_out will
      // all be false.
      const auto* buffer_allocation = iter->second;
      if (buffer_allocation->color() == memory_color &&
          !buffer_allocation->is_thread_local() &&
          !buffer_allocation->is_entry_computation_parameter() &&
          !buffer_allocation->is_constant() &&
          !buffer_allocation->maybe_live_out()) {
        return i;
      }
    }
  }
  return -1;
}

int64_t GetHeapSimulatorTraceId(const HloProto& proto, int64_t memory_color) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSconvertPShlo_proto_to_memory_visualization_utilsDTcc mht_17(mht_17_v, 863, "", "./tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.cc", "GetHeapSimulatorTraceId");

  int64_t id =
      GetHeapSimulatorTraceIdFromBufferAllocationIndex(proto, memory_color);
  if (id != -1) {
    return id;
  }
  return GetHeapSimulatorTraceIdFromEvents(proto, memory_color);
}

}  // namespace profiler
}  // namespace tensorflow
