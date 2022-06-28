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
class MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// This approach to arenas overcomes many of the limitations described
// in the "Specialized allocators" section of
//     http://www.pdos.lcs.mit.edu/~dm/c++-new.html
//
// A somewhat similar approach to Gladiator, but for heap-detection, was
// suggested by Ron van der Wal and Scott Meyers at
//     http://www.aristeia.com/BookErrata/M27Comments_frames.html

#include "tensorflow/core/lib/core/arena.h"

#include <assert.h>

#include <algorithm>
#include <vector>

#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"

namespace tensorflow {
namespace core {

// ----------------------------------------------------------------------
// Arena::Arena()
// Arena::~Arena()
//    Destroying the arena automatically calls Reset()
// ----------------------------------------------------------------------

Arena::Arena(const size_t block_size)
    : remaining_(0),
      block_size_(block_size),
      freestart_(nullptr),  // set for real in Reset()
      blocks_alloced_(1),
      overflow_blocks_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/lib/core/arena.cc", "Arena::Arena");

  assert(block_size > kDefaultAlignment);

  first_blocks_[0].mem =
      reinterpret_cast<char*>(port::AlignedMalloc(block_size_, sizeof(void*)));

  first_blocks_[0].size = block_size_;

  Reset();
}

Arena::~Arena() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/lib/core/arena.cc", "Arena::~Arena");

  FreeBlocks();
  assert(overflow_blocks_ == nullptr);  // FreeBlocks() should do that
  // The first X blocks stay allocated always by default.  Delete them now.
  for (size_t i = 0; i < blocks_alloced_; ++i) {
    port::AlignedFree(first_blocks_[i].mem);
  }
}

// Returns true iff it advances freestart_ to the first position
// satisfying alignment without exhausting the current block.
bool Arena::SatisfyAlignment(size_t alignment) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/lib/core/arena.cc", "Arena::SatisfyAlignment");

  const size_t overage = reinterpret_cast<size_t>(freestart_) & (alignment - 1);
  if (overage > 0) {
    const size_t waste = alignment - overage;
    if (waste >= remaining_) {
      return false;
    }
    freestart_ += waste;
    remaining_ -= waste;
  }
  DCHECK_EQ(size_t{0}, reinterpret_cast<size_t>(freestart_) & (alignment - 1));
  return true;
}

// ----------------------------------------------------------------------
// Arena::Reset()
//    Clears all the memory an arena is using.
// ----------------------------------------------------------------------

void Arena::Reset() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_3(mht_3_v, 269, "", "./tensorflow/core/lib/core/arena.cc", "Arena::Reset");

  FreeBlocks();
  freestart_ = first_blocks_[0].mem;
  remaining_ = first_blocks_[0].size;

  // There is no guarantee the first block is properly aligned, so
  // enforce that now.
  CHECK(SatisfyAlignment(kDefaultAlignment));

  freestart_when_empty_ = freestart_;
}

// ----------------------------------------------------------------------
// Arena::MakeNewBlock()
//    Our sbrk() equivalent.  We always make blocks of the same size
//    (though GetMemory() can also make a new block for really big
//    data.
// ----------------------------------------------------------------------

void Arena::MakeNewBlock(const uint32 alignment) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_4(mht_4_v, 291, "", "./tensorflow/core/lib/core/arena.cc", "Arena::MakeNewBlock");

  AllocatedBlock* block = AllocNewBlock(block_size_, alignment);
  freestart_ = block->mem;
  remaining_ = block->size;
  CHECK(SatisfyAlignment(alignment));
}

static uint32 LeastCommonMultiple(uint32 a, uint32 b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_5(mht_5_v, 301, "", "./tensorflow/core/lib/core/arena.cc", "LeastCommonMultiple");

  if (a > b) {
    return (a / MathUtil::GCD<uint32>(a, b)) * b;
  } else if (a < b) {
    return (b / MathUtil::GCD<uint32>(b, a)) * a;
  } else {
    return a;
  }
}

// -------------------------------------------------------------
// Arena::AllocNewBlock()
//    Adds and returns an AllocatedBlock.
//    The returned AllocatedBlock* is valid until the next call
//    to AllocNewBlock or Reset.  (i.e. anything that might
//    affect overflow_blocks_).
// -------------------------------------------------------------

Arena::AllocatedBlock* Arena::AllocNewBlock(const size_t block_size,
                                            const uint32 alignment) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_6(mht_6_v, 323, "", "./tensorflow/core/lib/core/arena.cc", "Arena::AllocNewBlock");

  AllocatedBlock* block;
  // Find the next block.
  if (blocks_alloced_ < TF_ARRAYSIZE(first_blocks_)) {
    // Use one of the pre-allocated blocks
    block = &first_blocks_[blocks_alloced_++];
  } else {  // oops, out of space, move to the vector
    if (overflow_blocks_ == nullptr)
      overflow_blocks_ = new std::vector<AllocatedBlock>;
    // Adds another block to the vector.
    overflow_blocks_->resize(overflow_blocks_->size() + 1);
    // block points to the last block of the vector.
    block = &overflow_blocks_->back();
  }

  // NOTE(tucker): this utility is made slightly more complex by
  // not disallowing the case where alignment > block_size.
  // Can we, without breaking existing code?

  // Must be a multiple of kDefaultAlignment, unless requested
  // alignment is 1, in which case we don't care at all.
  uint32 adjusted_alignment =
      (alignment > 1 ? LeastCommonMultiple(alignment, kDefaultAlignment) : 1);
  // Required minimum alignment for port::AlignedMalloc().
  adjusted_alignment =
      std::max(adjusted_alignment, static_cast<uint32>(sizeof(void*)));

  CHECK_LE(adjusted_alignment, static_cast<uint32>(1 << 20))
      << "Alignment on boundaries greater than 1MB not supported.";

  // If block_size > alignment we force block_size to be a multiple
  // of alignment; if block_size < alignment we make no adjustment.
  size_t adjusted_block_size = block_size;
  if (adjusted_block_size > adjusted_alignment) {
    const uint32 excess = adjusted_block_size % adjusted_alignment;
    adjusted_block_size += (excess > 0 ? adjusted_alignment - excess : 0);
  }
  block->mem = reinterpret_cast<char*>(
      port::AlignedMalloc(adjusted_block_size, adjusted_alignment));
  block->size = adjusted_block_size;
  CHECK(nullptr != block->mem) << "block_size=" << block_size
                               << " adjusted_block_size=" << adjusted_block_size
                               << " alignment=" << alignment
                               << " adjusted_alignment=" << adjusted_alignment;

  return block;
}

// ----------------------------------------------------------------------
// Arena::GetMemoryFallback()
//    We take memory out of our pool, aligned on the byte boundary
//    requested.  If we don't have space in our current pool, we
//    allocate a new block (wasting the remaining space in the
//    current block) and give you that.  If your memory needs are
//    too big for a single block, we make a special your-memory-only
//    allocation -- this is equivalent to not using the arena at all.
// ----------------------------------------------------------------------

void* Arena::GetMemoryFallback(const size_t size, const int alignment) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_7(mht_7_v, 384, "", "./tensorflow/core/lib/core/arena.cc", "Arena::GetMemoryFallback");

  if (0 == size) {
    return nullptr;  // stl/stl_alloc.h says this is okay
  }

  // alignment must be a positive power of 2.
  CHECK(alignment > 0 && 0 == (alignment & (alignment - 1)));

  // If the object is more than a quarter of the block size, allocate
  // it separately to avoid wasting too much space in leftover bytes.
  if (block_size_ == 0 || size > block_size_ / 4) {
    return AllocNewBlock(size, alignment)->mem;
  }

  // Enforce alignment on freestart_ then check for adequate space,
  // which may require starting a new block.
  if (!SatisfyAlignment(alignment) || size > remaining_) {
    MakeNewBlock(alignment);
  }
  CHECK_LE(size, remaining_);

  remaining_ -= size;
  void* result = freestart_;
  freestart_ += size;

  return result;
}

// ----------------------------------------------------------------------
// Arena::ReturnMemoryFallback()
// Arena::FreeBlocks()
//    Unlike GetMemory(), which does actual work, ReturnMemory() is a
//    no-op: we don't "free" memory until Reset() is called.  We do
//    update some stats, though.  Note we do no checking that the
//    pointer you pass in was actually allocated by us, or that it
//    was allocated for the size you say, so be careful here!
//       FreeBlocks() does the work for Reset(), actually freeing all
//    memory allocated in one fell swoop.
// ----------------------------------------------------------------------

void Arena::FreeBlocks() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSarenaDTcc mht_8(mht_8_v, 427, "", "./tensorflow/core/lib/core/arena.cc", "Arena::FreeBlocks");

  for (size_t i = 1; i < blocks_alloced_; ++i) {  // keep first block allocated
    port::AlignedFree(first_blocks_[i].mem);
    first_blocks_[i].mem = nullptr;
    first_blocks_[i].size = 0;
  }
  blocks_alloced_ = 1;
  if (overflow_blocks_ != nullptr) {
    std::vector<AllocatedBlock>::iterator it;
    for (it = overflow_blocks_->begin(); it != overflow_blocks_->end(); ++it) {
      port::AlignedFree(it->mem);
    }
    delete overflow_blocks_;  // These should be used very rarely
    overflow_blocks_ = nullptr;
  }
}

}  // namespace core
}  // namespace tensorflow
