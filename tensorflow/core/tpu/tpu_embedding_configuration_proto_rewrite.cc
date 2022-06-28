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
class MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc() {
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

#include "tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace {

// Validates that the batch_size_per_tensor_core and
// TableDescriptor.num_features fields have been populated correctly in the TPU
// embedding configuration.
Status ValidateBatchSizeAndFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "ValidateBatchSizeAndFeatureCounts");

  if (config.batch_size_per_tensor_core() <= 0) {
    return errors::InvalidArgument(absl::StrFormat(
        "Invalid batch_size_per_tensor_core: %d found in the TPU embedding "
        "configuration. Valid values are >0.",
        config.batch_size_per_tensor_core()));
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() <= 0) {
      return errors::InvalidArgument(absl::StrFormat(
          "Invalid num_features: %d found for table: %s in the TPU embedding "
          "configuration. Valid values are >0.",
          table_config.num_features(), table_config.name()));
    }
  }  // table_config
  return Status::OK();
}

// Validates that the batch_size_per_tensor_core and
// TableDescriptor.num_features fields are NOT populated in the TPU embedding
// configuration when the feature descriptor fields are filled in.
Status ValidateBatchSizeAndFeatureCountsAreEmpty(
    const tpu::TPUEmbeddingConfiguration& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "ValidateBatchSizeAndFeatureCountsAreEmpty");

  if (config.batch_size_per_tensor_core() != 0) {
    return errors::InvalidArgument(
        "Invalid TPU embedding configuration. The batch_size_per_tensor_core "
        "field must NOT be populated when the feature_descriptor fields are "
        "filled in.");
  }
  for (const auto& table_config : config.table_descriptor()) {
    if (table_config.num_features() != 0) {
      return errors::InvalidArgument(absl::StrFormat(
          "Invalid TPU embedding configuration. The "
          "TableDescriptor.num_features field must NOT be populated when the "
          "feature_descriptor fields are filled in, num_features is set to %d "
          "for table %s.",
          table_config.num_features(), table_config.name()));
    }
  }  // table_config
  return Status::OK();
}

// Validates that the feature_descriptor fields have been correctly filled in.
// All tables must have at least one input feature.
Status ValidateFeatureDescriptors(
    const tpu::TPUEmbeddingConfiguration& config) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "ValidateFeatureDescriptors");

  const int table_count = config.table_descriptor_size();
  std::vector<bool> tables_present(table_count, false);

  for (const auto& feature_config : config.feature_descriptor()) {
    const int table_id = feature_config.table_id();
    const auto& input_shape = feature_config.input_shape();
    if (table_id < 0 || table_id >= table_count) {
      return errors::InvalidArgument(absl::StrFormat(
          "Invalid table_id: %d found in feature_descriptor: %s, all table_ids "
          "must be in the range[0, %d)",
          table_id, feature_config.ShortDebugString(), table_count));
    }
    if (input_shape.empty()) {
      return errors::InvalidArgument(absl::StrFormat(
          "The input_shape field cannot be empty in feature_descriptor: %s",
          feature_config.ShortDebugString()));
    }
    for (const int dim_size : input_shape) {
      if (dim_size <= 0) {
        return errors::InvalidArgument(absl::StrFormat(
            "The input_shape dimension sizes must all be >0 in "
            "feature_descriptor: %s, found dimension size set to %d",
            feature_config.ShortDebugString(), dim_size));
      }
    }
    tables_present[table_id] = true;
  }  // feature_config

  for (int table_id = 0; table_id < table_count; ++table_id) {
    if (!tables_present[table_id]) {
      return errors::InvalidArgument(absl::StrFormat(
          "No feature_descriptor fields found for table: %s (ID: %d) in "
          "the TPU embedding configuration.",
          config.table_descriptor(table_id).name(), table_id));
    }
  }
  return Status::OK();
}

// Populates the feature_descriptor fields with default values when they have
// not been filled in by the user.
void PopulateFeatureDescriptors(tpu::TPUEmbeddingConfiguration* config) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_3(mht_3_v, 296, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "PopulateFeatureDescriptors");

  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    tpu::TPUEmbeddingConfiguration::FeatureDescriptor* feature_descriptor =
        config->add_feature_descriptor();
    feature_descriptor->set_table_id(table_id);
    feature_descriptor->add_input_shape(
        config->batch_size_per_tensor_core() *
        config->table_descriptor(table_id).num_features());
  }  // table_id
}

// Computes the input feature batch size based on the input feature shape. As
// we treat the last dimension as the reduction dimension, the batch size should
// be the product of all the axes except the last one.
std::vector<int> ComputeInputFeatureBatchSizes(
    const tpu::TPUEmbeddingConfiguration& config) {
  std::vector<int32> input_feature_batch_sizes;
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    const int32 batch_size =
        absl::c_accumulate(config.feature_descriptor(i).input_shape(),
                           /*init=*/1, std::multiplies<>());
    input_feature_batch_sizes.push_back(batch_size);
  }
  return input_feature_batch_sizes;
}

// Computes the TensorCore batch size as the GCD of all input feature batch
// sizes.
int ComputeBatchSizePerTensorCore(
    absl::Span<const int> input_feature_batch_sizes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_4(mht_4_v, 329, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "ComputeBatchSizePerTensorCore");

  uint32_t batch_size = input_feature_batch_sizes[0];
  for (const uint32_t input_feature_batch_size : input_feature_batch_sizes) {
    batch_size =
        tensorflow::MathUtil::GCD(batch_size, input_feature_batch_size);
  }
  return batch_size;
}

// Computes the TPU feature counts per user table as the sum of the TPU feature
// counts of the constituent input features. The TPU feature count for an input
// feature is the ratio of the batch size for that input feature to the batch
// size per TensorCore.
std::vector<int> ComputeTpuFeatureCounts(
    const tpu::TPUEmbeddingConfiguration& config,
    absl::Span<const int> input_feature_batch_sizes,
    int batch_size_per_tensor_core) {
  DCHECK_EQ(input_feature_batch_sizes.size(), config.feature_descriptor_size());
  std::vector<int> tpu_feature_counts(config.table_descriptor_size(), 0);
  for (int i = 0; i < config.feature_descriptor_size(); ++i) {
    DCHECK_EQ(input_feature_batch_sizes[i] % batch_size_per_tensor_core, 0);
    tpu_feature_counts[config.feature_descriptor(i).table_id()] +=
        (input_feature_batch_sizes[i] / batch_size_per_tensor_core);
  }
  return tpu_feature_counts;
}

// Populates default values for batch_size_per_tensor_core and
// TableDescriptor.num_features when they have not been filled in by the user.
// The batch_size_per_tensor_core is computed as the GCD of the batch sizes of
// all input features.
void PopulateBatchSizeAndFeatureCounts(tpu::TPUEmbeddingConfiguration* config) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_5(mht_5_v, 363, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "PopulateBatchSizeAndFeatureCounts");

  const std::vector<int> input_feature_batch_sizes =
      ComputeInputFeatureBatchSizes(*config);
  const int batch_size_per_tensor_core =
      ComputeBatchSizePerTensorCore(input_feature_batch_sizes);
  const std::vector<int> tpu_feature_counts = ComputeTpuFeatureCounts(
      *config, input_feature_batch_sizes, batch_size_per_tensor_core);
  config->set_batch_size_per_tensor_core(batch_size_per_tensor_core);
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    auto* table_config = config->mutable_table_descriptor(table_id);
    table_config->set_num_features(tpu_feature_counts[table_id]);
  }  // table_id
}

}  // namespace

Status PopulateMissingFieldsInTPUEmbeddingConfig(
    tpu::TPUEmbeddingConfiguration* config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPStpu_embedding_configuration_proto_rewriteDTcc mht_6(mht_6_v, 384, "", "./tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc", "PopulateMissingFieldsInTPUEmbeddingConfig");

  if (config->feature_descriptor_size() == 0) {
    // If the feature_descriptor list is empty, validate that the batch size and
    // feature counts have been set properly. then, populate the
    // feature_descriptor with appropriate values.
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCounts(*config));
    PopulateFeatureDescriptors(config);
  } else {
    // If the feature_descriptor list is non-empty, validate that the batch size
    // and feature counts have NOT been populated. Also, validate that the
    // feature descriptors have been set properly. Then, populate the batch size
    // and feature counts with appropriate values.
    TF_RETURN_IF_ERROR(ValidateBatchSizeAndFeatureCountsAreEmpty(*config));
    TF_RETURN_IF_ERROR(ValidateFeatureDescriptors(*config));
    PopulateBatchSizeAndFeatureCounts(config);
  }
  return Status::OK();
}

}  // namespace tensorflow
