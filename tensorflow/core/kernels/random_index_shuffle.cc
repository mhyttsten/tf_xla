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
class MHTracer_DTPStensorflowPScorePSkernelsPSrandom_index_shuffleDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_index_shuffleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSrandom_index_shuffleDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements random_index_shuffle() by using a simple block chiper
// for pseudorandom permutations.
// This idea is described as cycle-walking cipher in
// https://www.cs.ucdavis.edu/~rogaway/papers/subset.pdf
//
// We use the Simon block cipher described in
// https://eprint.iacr.org/2013/404
// and following recommendations in
// https://nsacyber.github.io/simon-speck/implementations/ImplementationGuide1.1.pdf.
// However we use a single fixed key size and support arbtitary block sizes.
// Further we fixed the number of rounds in the Feistel structuro to be always
// 4. This reduces the computational cost and still gives good shuffle behavior.
//
// Warning: Given the modifications descripted above this implementation should
// not be used for application that require cryptograhic secure RNGs.

#include "tensorflow/core/kernels/random_index_shuffle.h"

#include <assert.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {
// Some of the macros below require a minimum word size of 8
// (2 word size = block size).
// Smaller block sizes might give poor results in terms of randomness.
constexpr int kMinBlockSize = 16;

namespace impl {

#define ROTL(x, r, W) (((x) << (r)) | (x >> (W - (r))))
#define ROTR(x, r, W) (((x) >> (r)) | ((x) << (W - (r))))
#define SIMON_F(x, W) ((ROTL(x, 1, W) & ROTL(x, 8, W) ^ ROTL(x, 2, W)))
#define SIMON_Rx2(x, y, k1, k2, W) \
  (y ^= SIMON_F(x, W), y ^= k1, x ^= SIMON_F(y, W), x ^= k2)

// Returns the keys per round for a Simon cipher.
// This variant uses std::bitset and can generate keys with any number of bits.
// This should not be used to encrypt data. It's not secure. We only use it
// to generate pseudorandom permutations.
template <int W, int ROUNDS>
std::array<std::bitset<W>, ROUNDS> simon_key_schedule(
    const std::array<uint32_t, 3>& key) {
  // Required by ROTR/ROTL
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  const auto c = std::bitset<W>(0xfffffffc);
  auto z = std::bitset<W>(0x7369f885192c0ef5LL);
  std::array<std::bitset<W>, ROUNDS> rk;
  rk[0] = key[0];
  rk[1] = key[1];
  rk[2] = key[2];
  for (int i = 3; i < ROUNDS; i++) {
    rk[i] = c ^ (z & std::bitset<W>(1)) ^ rk[i - 3] ^ ROTR(rk[i - 1], 3, W) ^
            ROTR(rk[i - 1], 4, W);
    z >>= 1;
  }
  return rk;
}

// Encrypts the given value using the Simon chipher.
// This is should not be used to encrypt data. It's not secure. We only use it
// to generate pseudorandom permutations.
template <int W, int ROUNDS>
uint64_t simon_encrypt(const uint64_t value,
                       const std::array<std::bitset<W>, ROUNDS>& round_keys) {
  // Required by ROTR/ROTL
  static_assert(W >= 8, "Minimum word size is 8 bits.");
  std::bitset<W> left(value >> W);
  std::bitset<W> right(value);
  for (int i = 0; i < ROUNDS;) {
    SIMON_Rx2(right, left, round_keys[i++], round_keys[i++], W);
  }
  return (left.to_ullong() << W) | right.to_ullong();
}

// In the original implementation the number of rounds depends on the block
// size, key size and key words. For our purposes of random shuffle a 4 rounds
// is fine.
// W = word size
// B = 2 * W = block size
template <int B, int ROUNDS = 4>
uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index) {
  const auto round_keys = simon_key_schedule<B / 2, ROUNDS>(key);
  uint64_t new_index = index;
  while (true) {
    new_index = simon_encrypt<B / 2, ROUNDS>(new_index, round_keys);
    if (new_index <= max_index) {
      return new_index;
    }
  }
}

#undef ROTL
#undef ROTR
#undef SIMON_F
#undef SIMON_RxC

}  // namespace impl

uint64_t index_shuffle(const uint64_t index, const std::array<uint32_t, 3>& key,
                       const uint64_t max_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSrandom_index_shuffleDTcc mht_0(mht_0_v, 292, "", "./tensorflow/core/kernels/random_index_shuffle.cc", "index_shuffle");

  int block_size = static_cast<int>(std::ceil(std::log2(max_index)));
  block_size = std::max(block_size + block_size % 2, kMinBlockSize);
  assert(block_size > 0 && block_size % 2 == 0 && block_size <= 64);
  switch (block_size) {
    case 16:
      return impl::index_shuffle<16>(index, key, max_index);
    case 18:
      return impl::index_shuffle<18>(index, key, max_index);
    case 20:
      return impl::index_shuffle<20>(index, key, max_index);
    case 22:
      return impl::index_shuffle<22>(index, key, max_index);
    case 24:
      return impl::index_shuffle<24>(index, key, max_index);
    case 26:
      return impl::index_shuffle<26>(index, key, max_index);
    case 28:
      return impl::index_shuffle<28>(index, key, max_index);
    case 30:
      return impl::index_shuffle<30>(index, key, max_index);
    case 32:
      return impl::index_shuffle<32>(index, key, max_index);
    case 34:
      return impl::index_shuffle<34>(index, key, max_index);
    case 36:
      return impl::index_shuffle<36>(index, key, max_index);
    case 38:
      return impl::index_shuffle<38>(index, key, max_index);
    case 40:
      return impl::index_shuffle<40>(index, key, max_index);
    case 42:
      return impl::index_shuffle<42>(index, key, max_index);
    case 44:
      return impl::index_shuffle<44>(index, key, max_index);
    case 46:
      return impl::index_shuffle<46>(index, key, max_index);
    case 48:
      return impl::index_shuffle<48>(index, key, max_index);
    case 50:
      return impl::index_shuffle<50>(index, key, max_index);
    case 52:
      return impl::index_shuffle<52>(index, key, max_index);
    case 54:
      return impl::index_shuffle<54>(index, key, max_index);
    case 56:
      return impl::index_shuffle<56>(index, key, max_index);
    case 58:
      return impl::index_shuffle<58>(index, key, max_index);
    case 60:
      return impl::index_shuffle<60>(index, key, max_index);
    case 62:
      return impl::index_shuffle<62>(index, key, max_index);
    default:
      return impl::index_shuffle<64>(index, key, max_index);
  }
}

}  // namespace random
}  // namespace tensorflow
