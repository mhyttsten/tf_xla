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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc() {
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
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/kernels/summary_interface.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/events_writer.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

static mutex* get_counters_map_lock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "get_counters_map_lock");

  static mutex counters_map_lock(LINKER_INITIALIZED);
  return &counters_map_lock;
}

static std::unordered_map<string, monitoring::Counter<1>*>* get_counters_map() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "get_counters_map");

  static std::unordered_map<string, monitoring::Counter<1>*>* counters_map =
      new std::unordered_map<string, monitoring::Counter<1>*>;
  return counters_map;
}

class StatsAggregatorImpl : public StatsAggregator {
 public:
  StatsAggregatorImpl() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorImpl");
}

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      const int64_t steps) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddToHistogram");

    mutex_lock l(mu_);
    histogram::Histogram& histogram = histograms_[name];
    for (double value : values) {
      histogram.Add(value);
    }
  }

  void AddScalar(const string& name, float value,
                 const int64_t steps) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddScalar");

    mutex_lock l(mu_);
    scalars_[name] = value;
  }

  void EncodeToProto(Summary* out_summary) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "EncodeToProto");

    mutex_lock l(mu_);
    for (const auto& pair : histograms_) {
      const string& name = pair.first;
      const histogram::Histogram& histogram = pair.second;

      Summary::Value* value = out_summary->add_value();
      value->set_tag(name);
      histogram.EncodeToProto(value->mutable_histo(),
                              false /* doesn't preserve zero buckets */);
    }
    for (const auto& pair : scalars_) {
      Summary::Value* value = out_summary->add_value();
      value->set_tag(pair.first);
      value->set_simple_value(pair.second);
    }
  }

  // StatsAggregator implementation for V2 is based on push-based summary, no-op
  // in V1.
  Status SetSummaryWriter(
      SummaryWriterInterface* summary_writer_interface) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "SetSummaryWriter");

    return Status::OK();
  }

  void IncrementCounter(const string& name, const string& label,
                        int64_t val) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   mht_7_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_7(mht_7_v, 286, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "IncrementCounter");

    mutex_lock l(*get_counters_map_lock());
    auto counters_map = get_counters_map();
    if (counters_map->find(name) == counters_map->end()) {
      counters_map->emplace(
          name,
          monitoring::Counter<1>::New(
              /*streamz name*/ name,
              /*streamz description*/
              strings::StrCat(name, " generated or consumed by the component."),
              /*streamz label name*/ "component_descriptor"));
    }
    counters_map->at(name)->GetCell(label)->IncrementBy(val);
  }

 private:
  mutex mu_;
  std::unordered_map<string, histogram::Histogram> histograms_
      TF_GUARDED_BY(mu_);
  std::unordered_map<string, float> scalars_ TF_GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorImpl);
};

class StatsAggregatorHandleOp
    : public ResourceOpKernel<StatsAggregatorResource> {
 public:
  explicit StatsAggregatorHandleOp(OpKernelConstruction* ctx)
      : ResourceOpKernel<StatsAggregatorResource>(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorHandleOp");
}

 private:
  Status CreateResource(StatsAggregatorResource** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret =
        new StatsAggregatorResource(absl::make_unique<StatsAggregatorImpl>());
    return Status::OK();
  }
};

class StatsAggregatorImplV2 : public StatsAggregator {
 public:
  StatsAggregatorImplV2() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorImplV2");
}

  ~StatsAggregatorImplV2() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_10(mht_10_v, 337, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "~StatsAggregatorImplV2");

    if (summary_writer_interface_) {
      summary_writer_interface_->Unref();
    }
  }

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      const int64_t steps) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_11(mht_11_v, 348, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddToHistogram");

    mutex_lock l(mu_);
    histogram::Histogram& histogram = histograms_[name];
    for (double value : values) {
      histogram.Add(value);
    }
    AddToEvents(name, steps, histogram);
  }

  void AddScalar(const string& name, float value,
                 const int64_t steps) override {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_12(mht_12_v, 362, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddScalar");

    mutex_lock l(mu_);
    AddToEvents(name, steps, value);
  }

  // TODO(b/116314787): expose this is public API to manually flush summary.
  Status Flush() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_13(mht_13_v, 371, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "Flush");

    mutex_lock l(mu_);
    if (summary_writer_interface_)
      TF_RETURN_IF_ERROR(summary_writer_interface_->Flush());
    return Status::OK();
  }

  void IncrementCounter(const string& name, const string& label,
                        int64_t val) override {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + name + "\"");
   mht_14_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_14(mht_14_v, 384, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "IncrementCounter");

    mutex_lock l(*get_counters_map_lock());
    auto counters_map = get_counters_map();
    if (counters_map->find(name) == counters_map->end()) {
      counters_map->emplace(
          name, monitoring::Counter<1>::New(
                    /*streamz name*/ "/tensorflow/" + name,
                    /*streamz description*/
                    name + " generated or consumed by the component.",
                    /*streamz label name*/ "component_descriptor"));
    }
    counters_map->at(name)->GetCell(label)->IncrementBy(val);
  }

  // StatsAggregator implementation for V1 is based on pull-based summary, no-op
  // in V2.
  void EncodeToProto(Summary* out_summary) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_15(mht_15_v, 403, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "EncodeToProto");
}

  Status SetSummaryWriter(
      SummaryWriterInterface* summary_writer_interface) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_16(mht_16_v, 409, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "SetSummaryWriter");

    mutex_lock l(mu_);
    if (summary_writer_interface_) {
      summary_writer_interface_->Unref();
      // If we create stats_aggregator twice in a program, we would end up with
      // already existing resource. In this case emitting an error if a
      // `summary_writer_resource` is present is not the intended behavior, we
      // could either Unref the existing summary_writer_resource or not set the
      // new resource at all.
    }
    summary_writer_interface_ = summary_writer_interface;
    summary_writer_interface_->Ref();
    return Status::OK();
  }

 private:
  void AddToEvents(const string& name, const int64_t steps,
                   const float scalar_value) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_17(mht_17_v, 430, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddToEvents");

    if (summary_writer_interface_ == nullptr) {
      return;
    }
    std::unique_ptr<Event> e{new Event};
    e->set_step(steps);
    e->set_wall_time(EnvTime::NowMicros() / 1.0e6);
    // maybe expose GetWallTime in SummaryWriterInterface
    Summary::Value* v = e->mutable_summary()->add_value();
    v->set_tag(name);
    v->set_simple_value(scalar_value);
    TF_CHECK_OK(summary_writer_interface_->WriteEvent(std::move(e)));
  }

  void AddToEvents(const string& name, const int64_t steps,
                   const histogram::Histogram& histogram)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_18(mht_18_v, 450, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "AddToEvents");

    if (summary_writer_interface_ == nullptr) {
      return;
    }
    std::unique_ptr<Event> e{new Event};
    e->set_step(steps);
    e->set_wall_time(EnvTime::NowMicros() / 1.0e6);
    Summary::Value* v = e->mutable_summary()->add_value();
    v->set_tag(name);
    histogram.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);
    TF_CHECK_OK(summary_writer_interface_->WriteEvent(std::move(e)));
  }

  mutex mu_;
  SummaryWriterInterface* summary_writer_interface_ TF_GUARDED_BY(mu_) =
      nullptr;
  // not owned, we might be associating the default summary_writer from the
  // context
  std::unordered_map<string, histogram::Histogram> histograms_
      TF_GUARDED_BY(mu_);
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorImplV2);
};

class StatsAggregatorHandleOpV2
    : public ResourceOpKernel<StatsAggregatorResource> {
 public:
  explicit StatsAggregatorHandleOpV2(OpKernelConstruction* ctx)
      : ResourceOpKernel<StatsAggregatorResource>(ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_19(mht_19_v, 480, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorHandleOpV2");
}

 private:
  Status CreateResource(StatsAggregatorResource** ret) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret =
        new StatsAggregatorResource(absl::make_unique<StatsAggregatorImplV2>());
    return Status::OK();
  }
};

class StatsAggregatorSummaryOp : public OpKernel {
 public:
  explicit StatsAggregatorSummaryOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_20(mht_20_v, 497, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorSummaryOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_21(mht_21_v, 502, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "Compute");

    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    Tensor* summary_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &summary_t));
    Summary summary;
    resource->stats_aggregator()->EncodeToProto(&summary);
    summary_t->scalar<tstring>()() = summary.SerializeAsString();
  }
};

class StatsAggregatorSetSummaryWriterOp : public OpKernel {
 public:
  explicit StatsAggregatorSetSummaryWriterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_22(mht_22_v, 525, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "StatsAggregatorSetSummaryWriterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_aggregator_opsDTcc mht_23(mht_23_v, 530, "", "./tensorflow/core/kernels/data/experimental/stats_aggregator_ops.cc", "Compute");

    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    const Tensor& summary_resource_handle_t = ctx->input(1);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(summary_resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));
    core::RefCountPtr<SummaryWriterInterface> summary_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &summary_resource));
    TF_CHECK_OK(
        resource->stats_aggregator()->SetSummaryWriter(summary_resource.get()));
  }
};

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorHandle").Device(DEVICE_CPU),
                        StatsAggregatorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalStatsAggregatorHandle").Device(DEVICE_CPU),
    StatsAggregatorHandleOp);

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorHandleV2").Device(DEVICE_CPU),
                        StatsAggregatorHandleOpV2);

REGISTER_KERNEL_BUILDER(Name("StatsAggregatorSummary").Device(DEVICE_CPU),
                        StatsAggregatorSummaryOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalStatsAggregatorSummary").Device(DEVICE_CPU),
    StatsAggregatorSummaryOp);

REGISTER_KERNEL_BUILDER(
    Name("StatsAggregatorSetSummaryWriter").Device(DEVICE_CPU),
    StatsAggregatorSetSummaryWriterOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
