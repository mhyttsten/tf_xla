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
class MHTracer_DTPStensorflowPScorePSkernelsPSfractional_pool_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_pool_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfractional_pool_commonDTcc() {
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
#include "tensorflow/core/kernels/fractional_pool_common.h"

#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
static std::vector<int64_t> GeneratePoolingSequencePseudoRandom(
    int input_length, int output_length, GuardedPhiloxRandom* generator) {
  std::vector<int64_t> cum_seq(output_length + 1, 0);
  std::vector<int64_t> diff(output_length, 0);

  double alpha = static_cast<double>(input_length) / output_length;
  int k = input_length / output_length;

  // In the paper [1], author proposes the following procedure to generate a
  // pseudo random pooling region:
  //   1) Set a_0 = 1, a_Nout = Nin;
  //   2) a_i = ceil(alpha*(u+i))
  //      in which, i = 1, 2, ... , Nout-1
  //                u is a random number in (0,1) for all i
  //                alpha = Nin/Nout in (1,2)
  // The sequence {a_i} should satisfy a_i-a_{i-1} = 1 or 2
  // Note: for step 1), it makes more sense to make a_Nout = Nin+1, that way,
  //    a_i-a_{i-1} = 1 or 2 is also true for i = Nout.
  //
  // However, there are choices of alpha and u that will make
  // a_i - a_{i-1} > 2. This happens at the left boundary. For example, with
  // alpha = 1.732, u = 0.8, then a_1 = 4, a_1-a_0 = 3.
  // This is why u_max1 is needed, i.e. u is a random number in (0,u_max1)
  // instead of (0,1).
  // Define k = ceil(alpha)-1, then we require:
  //   a_1 = alpha*(u+1) <= a_0+(k+1)
  // ===> This gives u_max1 = (k+2)/alpha - 1.
  //
  // In addition, when extending the pooling sequence generation process for
  // alpha beyond (1,2), e.g. (k,k+1); a check on the right boundary is also
  // needed to make sure the last gap a_Nout-a_{Nout-1} >= k. Solving it gives
  // u_max2 = (Nin+1-k)/alpha - (Nout-1)
  // Here is an example where u > u_max2, alpha = 2.3, u = 0.7, u_max2 = 0.565,
  // Nin = 23, Nout = 10; the sequence
  // from a_0 to a_10 is:
  // [1, 4, 7, 9, 11, 14, 16, 18, 21, 23, 24]
  // The last gap is only 1.
  //
  // [1]: https://arxiv.org/abs/1412.6071
  double u_max1 = (k + 2) / alpha - 1;
  double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
  double max_u = std::min(u_max1, u_max2);

  // Generate random number in parallel.
  auto local_gen = generator->ReserveSamples32(2);
  random::SimplePhilox random(&local_gen);
  const double u = random.RandDouble() * max_u;

  cum_seq[0] = 1;
  cum_seq[output_length] = input_length + 1;
  for (int i = 1; i < output_length; ++i) {
    cum_seq[i] = static_cast<int>(ceil(alpha * (i + u)));
  }

  for (int i = 0; i < output_length; ++i) {
    diff[i] = cum_seq[i + 1] - cum_seq[i];
  }

  return diff;
}

static std::vector<int64_t> GeneratePoolingSequenceRandom(
    int input_length, int output_length, GuardedPhiloxRandom* generator) {
  int k = input_length / output_length;
  int num_random_spot = input_length % output_length;
  std::vector<int64_t> diff(output_length, k);

  for (int i = 0; i < num_random_spot; ++i) {
    diff[i] += 1;
  }

  // Randomly shuffle this vector.
  auto local_gen = generator->ReserveSamples32(diff.size());
  random::SingleSampleAdapter<random::PhiloxRandom> single(&local_gen);
  const auto uniform = [&single](uint32 n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfractional_pool_commonDTcc mht_0(mht_0_v, 263, "", "./tensorflow/core/kernels/fractional_pool_common.cc", "lambda");
 return single() % n; };
  RandomShuffle(diff.begin(), diff.end(), uniform);

  return diff;
}

std::vector<int64_t> GeneratePoolingSequence(int input_length,
                                             int output_length,
                                             GuardedPhiloxRandom* generator,
                                             bool pseudo_random) {
  std::vector<int64_t> diff;
  // This is a case that regular pooling can handle, just return diff with
  // each element input_length/output_length.
  if (input_length % output_length == 0) {
    diff = std::vector<int64_t>(output_length, input_length / output_length);
  }

  if (pseudo_random) {
    diff = GeneratePoolingSequencePseudoRandom(input_length, output_length,
                                               generator);
  } else {
    diff =
        GeneratePoolingSequenceRandom(input_length, output_length, generator);
  }

  // Sanity check.
  int k = input_length / output_length;
  for (int i = 0; i < output_length; ++i) {
    // k<= diff[i] <= k+1.
    DCHECK_GE(diff[i], k);
    DCHECK_LE(diff[i], k + 1);
  }

  // Return cumulative sequence.
  std::vector<int64_t> cum_seq(output_length + 1, 0);
  for (int i = 1; i < cum_seq.size(); ++i) {
    cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
  }
  return cum_seq;
}

}  // namespace tensorflow
