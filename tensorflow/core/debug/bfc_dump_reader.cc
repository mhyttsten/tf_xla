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
class MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc {
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
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc() {
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
#include <cinttypes>
#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
MemoryDump ReadDumpFile(const string& fname) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "ReadDumpFile");

  Status status;
  uint64 file_size = 0;
  status = Env::Default()->GetFileSize(fname, &file_size);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to get size of " << fname;
    exit(1);
  }
  std::unique_ptr<RandomAccessFile> file;
  status = Env::Default()->NewRandomAccessFile(fname, &file);
  if (!status.ok()) {
    LOG(ERROR) << "read from file " << fname << " failed " << status;
  }
  std::unique_ptr<char> buffer(static_cast<char*>(malloc(file_size + 1)));
  DCHECK(buffer.get());
  StringPiece contents(buffer.get(), file_size);
  status = file->Read(0, file_size, &contents, buffer.get());
  if (!status.ok()) {
    LOG(ERROR) << "read from file " << fname << " failed " << status;
  }
  MemoryDump md;
  md.ParseFromString(string(contents));
  return md;
}

MemoryDump FilterByChunkType(MemoryDump md, const char chunk_type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("chunk_type: '" + std::string(1, chunk_type) + "'");
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "FilterByChunkType");

  if (chunk_type == 'A' || (chunk_type != 'U' && chunk_type != 'F')) {
    return md;
  }
  MemoryDump filtered;
  filtered.set_allocator_name(md.allocator_name());
  for (const auto& it : md.bin_summary()) {
    *filtered.add_bin_summary() = it;
  }
  for (const auto& it : md.chunk()) {
    if ((chunk_type == 'U' && it.in_use()) ||
        (chunk_type == 'F' && !it.in_use())) {
      *filtered.add_chunk() = it;
    }
  }
  return filtered;
}

void PrintChunk(const MemChunk& mc, const uint64 ac_offset, bool freed_at,
                const int64_t total_bytes, int64_t* cumulative_bytes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintChunk");

  // A size class corresponding approximately to log base 100.
  int size_class = floor(0.5 * log10(static_cast<double>(mc.size())));
  *cumulative_bytes += mc.size();
  printf("  %c %d %p bin=%d bytes=%" PRIu64 " %3.1f%%", mc.in_use() ? 'U' : 'F',
         size_class, reinterpret_cast<const void*>(mc.address()), mc.bin(),
         static_cast<uint64_t>(mc.size()),
         100 * (*cumulative_bytes / static_cast<float>(total_bytes)));
  if (freed_at) {
    printf(" freed_at=%" PRIu64, static_cast<uint64_t>(mc.freed_at_count()));
  }
  if (ac_offset > 0) {
    printf(" age=%" PRIu64,
           static_cast<uint64_t>(ac_offset - mc.action_count()));
  } else {
    printf(" ac=%" PRIu64, static_cast<uint64_t>(mc.action_count()));
  }
  // step_ids are random, so save space by showing only low 16 bits.
  printf(" step=%x op=%s\n", static_cast<uint>(0xFFFF & mc.step_id()),
         mc.op_name().c_str());
}

void PrintSummary(const MemoryDump& md) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintSummary");

  printf("MemoryMap for allocator %s\n", md.allocator_name().c_str());
  for (auto& it : md.bin_summary()) {
    printf("   Bin %2d total bytes=%10" PRId64 " \tin use=%10" PRId64
           " \ttotal_chunks=%6" PRId64
           " "
           "\tin_use=%6" PRId64 "\n",
           it.bin(), static_cast<int64_t>(it.total_bytes_in_bin()),
           static_cast<int64_t>(it.total_bytes_in_use()),
           static_cast<int64_t>(it.total_chunks_in_bin()),
           static_cast<int64_t>(it.total_chunks_in_use()));
  }
  printf("Total num_allocs: %" PRId64 ", bytes_in_use: %" PRId64
         ", peak_bytes_in_use: %" PRId64
         ",\n"
         "largest_alloc_size: %" PRId64 ", fragmentation: %f\n",
         static_cast<int64_t>(md.stats().num_allocs()),
         static_cast<int64_t>(md.stats().bytes_in_use()),
         static_cast<int64_t>(md.stats().peak_bytes_in_use()),
         static_cast<int64_t>(md.stats().largest_alloc_size()),
         md.stats().fragmentation_metric());
}

void PrintSortedChunks(
    const MemoryDump& md,
    std::function<bool(const MemChunk*, const MemChunk*)> compare, bool by_age,
    bool freed_at, bool by_addr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_4(mht_4_v, 301, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintSortedChunks");

  std::vector<const MemChunk*> chunks;
  chunks.reserve(md.chunk_size());
  int64_t total_bytes = 0;
  int64_t cumulative_bytes = 0;
  uint64 max_action_count = 0;
  for (auto& it : md.chunk()) {
    chunks.push_back(&it);
    total_bytes += it.size();
    if (by_age && it.action_count() > max_action_count) {
      max_action_count = it.action_count();
    }
  }
  sort(chunks.begin(), chunks.end(), compare);
  uint64 last_end = 0;
  for (int i = 0; i < chunks.size(); ++i) {
    const MemChunk* c = chunks[i];
    if (by_addr && i > 0 && last_end != c->address()) {
      printf("  empty range from %p to %p  (%" PRId64 ")\n",
             reinterpret_cast<const void*>(last_end),
             reinterpret_cast<const void*>(c->address()),
             static_cast<int64_t>(c->address() - last_end));
    }
    PrintChunk(*c, max_action_count, freed_at, total_bytes, &cumulative_bytes);
    last_end = c->address() + c->size();
  }
}

void PrintChunksByAddress(const MemoryDump& md, bool by_age, bool freed_at) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_5(mht_5_v, 332, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintChunksByAddress");

  printf("---------------Chunks by address:--------------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->address() < b->address();
      },
      by_age, freed_at, true /*by_addr*/);
}

void PrintChunksByActionCount(const MemoryDump& md, bool by_age,
                              bool freed_at) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_6(mht_6_v, 346, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintChunksByActionCount");

  printf("------------Chunks by last action count:----------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->action_count() < b->action_count();
      },
      by_age, freed_at, false /*by_addr*/);
}

void PrintChunksBySize(const MemoryDump& md, bool by_age, bool freed_at) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_7(mht_7_v, 359, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintChunksBySize");

  printf("------------Chunks by size:----------------------\n");
  PrintSortedChunks(
      md,
      [](const MemChunk* a, const MemChunk* b) {
        return a->size() > b->size();
      },
      by_age, freed_at, false /*by_addr*/);
}

void PrintChunksByOpName(const MemoryDump& md, const string& op_name,
                         bool by_age, bool freed_at) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_8(mht_8_v, 374, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintChunksByOpName");

  printf("------------Chunks matching \"%s\":----------------------\n",
         op_name.c_str());
  MemoryDump filtered;
  uint64 total_bytes = 0;
  filtered.set_allocator_name(md.allocator_name());
  for (const auto& it : md.bin_summary()) {
    *filtered.add_bin_summary() = it;
  }
  for (const auto& it : md.chunk()) {
    if (RE2::PartialMatch(it.op_name(), op_name)) {
      *filtered.add_chunk() = it;
      total_bytes += it.size();
    }
  }
  printf("\t%d matching Chunks of total size %" PRIu64 " bytes:\n",
         filtered.chunk_size(), static_cast<uint64_t>(total_bytes));
  PrintSortedChunks(
      filtered,
      [](const MemChunk* a, const MemChunk* b) {
        return a->address() > b->address();
      },
      by_age, freed_at, false /*by_addr*/);
}

void PrintSizeHistory(const MemoryDump& md, bool by_age) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_9(mht_9_v, 402, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "PrintSizeHistory");

  printf("------------Allocated Bytes by Action Count--------\n");
  printf("num snapshots: %d\n", md.snap_shot_size());
  uint64 max_action_count = 0;
  if (by_age) {
    for (auto& it : md.snap_shot()) {
      if (it.action_count() > max_action_count) {
        max_action_count = it.action_count();
      }
    }
  }
  for (auto& it : md.snap_shot()) {
    if (by_age) {
      printf("\tage=%" PRIu64 ", size=%" PRId64 "\n",
             static_cast<uint64_t>(max_action_count - it.action_count()),
             static_cast<int64_t>(it.size()));
    } else {
      printf("\tac=%" PRIu64 ", size=%" PRId64 "\n",
             static_cast<uint64_t>(it.action_count()),
             static_cast<int64_t>(it.size()));
    }
  }
}
}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdebugPSbfc_dump_readerDTcc mht_10(mht_10_v, 430, "", "./tensorflow/core/debug/bfc_dump_reader.cc", "main");

  std::string path = "";
  bool by_addr = false;
  bool by_ac = false;
  bool by_size = false;
  bool by_age = true;
  bool freed_at = false;
  bool size_history = false;
  std::string chunk_type = "A";
  std::string op_name = "";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("path", &path,
                       "Path of GPU BFCAllocator memory dump file"),
      tensorflow::Flag("by_addr", &by_addr,
                       "Whether to print Chunks by memory address"),
      tensorflow::Flag("by_ac", &by_ac,
                       "Whether to print Chunks by action count"),
      tensorflow::Flag("by_size", &by_size,
                       "Whether to print Chunks by decreasing size"),
      tensorflow::Flag("by_age", &by_age,
                       "If true, replace absolute action count with age "
                       "(max_action_count - action_count) in display."),
      tensorflow::Flag("op_name", &op_name,
                       "Whether to print only chunks whose Op name matches "
                       "the supplied regexp."),
      tensorflow::Flag("freed_at", &freed_at,
                       "Whether to display the freed_at value (only relevant "
                       "with timestamped allocator)."),
      tensorflow::Flag("chunk_type", &chunk_type,
                       "One of {'U', 'F', 'A'} meaning in Use, Free, or All "
                       "(default).  Displays only Chunks of this type."),
      tensorflow::Flag("size_history", &size_history,
                       "If true, show the size history."),
  };
  bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || path.empty()) {
    std::cerr << tensorflow::Flags::Usage(argv[0], flag_list);
    return -1;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::MemoryDump md = tensorflow::ReadDumpFile(path);
  tensorflow::PrintSummary(md);
  if (chunk_type != "A") {
    md = FilterByChunkType(md, chunk_type[0]);
  }
  if (by_addr) tensorflow::PrintChunksByAddress(md, by_age, freed_at);
  if (by_ac) tensorflow::PrintChunksByActionCount(md, by_age, freed_at);
  if (by_size) tensorflow::PrintChunksBySize(md, by_age, freed_at);
  if (!op_name.empty()) {
    tensorflow::PrintChunksByOpName(md, op_name, by_age, freed_at);
  }
  if (size_history) tensorflow::PrintSizeHistory(md, by_age);
}
