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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh() {
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


#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {

// Represents a pair of random seeds. By TensorFlow convention, if both seeds
// are 0, then pseudo-random values are used instead.
class RandomSeeds {
 public:
  RandomSeeds(int64_t seed, int64_t seed2)
      : input_seed_(seed),
        input_seed2_(seed2),
        seed_((seed | seed2) == 0 ? random::New64() : seed),
        seed2_((seed | seed2) == 0 ? random::New64() : seed2) {}

  int64_t input_seed() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "input_seed");
 return input_seed_; }
  int64_t input_seed2() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "input_seed2");
 return input_seed2_; }
  int64_t seed() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_2(mht_2_v, 215, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed");
 return seed_; }
  int64_t seed2() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_3(mht_3_v, 219, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed2");
 return seed2_; }

 private:
  const int64_t input_seed_;
  const int64_t input_seed2_;
  const int64_t seed_;
  const int64_t seed2_;
};

// Base class for seed generator resources. Subclasses customize how seeds are
// generated.
class SeedGenerator {
 public:
  virtual ~SeedGenerator() {}

  virtual int64_t seed() const = 0;
  virtual int64_t seed2() const = 0;
  virtual bool reshuffle_each_iteration() const = 0;

  virtual void GenerateSeeds(int64_t* seed1, int64_t* seed2) = 0;
  virtual void Reset() = 0;

  virtual int64_t num_random_samples() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_4(mht_4_v, 244, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "num_random_samples");

    tf_shared_lock l(mu_);
    return num_random_samples_;
  }
  virtual void set_num_random_samples(int64_t num_random_samples) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_5(mht_5_v, 251, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "set_num_random_samples");

    mutex_lock l(mu_);
    num_random_samples_ = num_random_samples;
  }

 protected:
  mutable mutex mu_;
  int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
};

// A resource wrapping a shared instance of a seed generator.
class SeedGeneratorManager : public ResourceBase {
 public:
  explicit SeedGeneratorManager(SeedGenerator* seed_generator)
      : seed_generator_(seed_generator) {}

  std::string DebugString() const override;

  std::shared_ptr<SeedGenerator> get() { return seed_generator_; }

 private:
  std::shared_ptr<SeedGenerator> seed_generator_;
};

// Always generates the specified seed values.
class FixedSeedGenerator : public SeedGenerator {
 public:
  explicit FixedSeedGenerator(RandomSeeds seeds) : seeds_(std::move(seeds)) {}

  int64_t seed() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_6(mht_6_v, 283, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed");
 return seeds_.seed(); }
  int64_t seed2() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_7(mht_7_v, 287, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed2");
 return seeds_.seed(); }
  bool reshuffle_each_iteration() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_8(mht_8_v, 291, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "reshuffle_each_iteration");
 return false; }

  void GenerateSeeds(int64_t* seed1, int64_t* seed2) override;
  void Reset() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_9(mht_9_v, 297, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "Reset");
}

 private:
  const RandomSeeds seeds_;
};

// Generates different (but deterministically chosen) seed values.
class RandomSeedGenerator : public SeedGenerator {
 public:
  explicit RandomSeedGenerator(RandomSeeds seeds)
      : seeds_(std::move(seeds)),
        parent_generator_(seeds_.seed(), seeds_.seed2()),
        generator_(&parent_generator_) {}

  int64_t seed() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_10(mht_10_v, 314, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed");
 return seeds_.seed(); }
  int64_t seed2() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_11(mht_11_v, 318, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "seed2");
 return seeds_.seed2(); }
  bool reshuffle_each_iteration() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrandom_seed_opsDTh mht_12(mht_12_v, 322, "", "./tensorflow/core/kernels/data/random_seed_ops.h", "reshuffle_each_iteration");
 return true; }

  void GenerateSeeds(int64_t* seed1, int64_t* seed2) override;
  void Reset() override;

 private:
  const RandomSeeds seeds_;
  random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
  random::SingleSampleAdapter<random::PhiloxRandom> generator_
      TF_GUARDED_BY(mu_);
};

// Creates an instance of seed generator resource and transfers ownership
// to the caller.
class AnonymousSeedGeneratorHandleOp
    : public AnonymousResourceOp<SeedGeneratorManager> {
 public:
  explicit AnonymousSeedGeneratorHandleOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string name() override;
  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        SeedGeneratorManager** manager) override;

  mutex mu_;
  std::unique_ptr<RandomSeeds> seeds_ TF_GUARDED_BY(mu_);
  bool reshuffle_;
};

// Deletes an instance of seed generator resource.
class DeleteSeedGeneratorOp : public OpKernel {
 public:
  explicit DeleteSeedGeneratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_RANDOM_SEED_OPS_H_
