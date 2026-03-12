/*
 * Minimal C++ runner for nanoQwen3.5MoE .pte files.
 *
 * Runs autoregressive text generation: feeds one token at a time,
 * picks the argmax, prints the character. No tokenizer needed —
 * the model uses char-level vocab (65 characters).
 *
 * Usage:
 *   ./runner --model_path nano_qwen35_moe_portable.pte
 *   ./runner --model_path nano_qwen35_moe_cuda.pte --data_path aoti_cuda_blob.ptd
 *   ./runner --model_path nano_qwen35_moe_cuda.pte --data_path aoti_cuda_blob.ptd --num_tokens 100
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

DEFINE_string(model_path, "", "Path to .pte file");
DEFINE_string(data_path, "", "Path to .ptd file (for CUDA delegate)");
DEFINE_string(prompt, "\nFirst Citizen:\n", "Prompt text");
DEFINE_int32(num_tokens, 50, "Number of tokens to generate");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy)");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

// Shakespeare char-level vocab (65 chars, same as nanoGPT)
// Index 0 = '\n', then sorted unique chars from tinyshakespeare
static const char VOCAB[] =
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    fprintf(stderr, "Usage: %s --model_path <.pte> [--data_path <.ptd>]\n",
            argv[0]);
    return 1;
  }

  // Load model
  printf("Loading: %s\n", FLAGS_model_path.c_str());
  std::unique_ptr<Module> module;
  if (!FLAGS_data_path.empty()) {
    printf("Data: %s\n", FLAGS_data_path.c_str());
    module = std::make_unique<Module>(
        FLAGS_model_path, FLAGS_data_path,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
  } else {
    module = std::make_unique<Module>(
        FLAGS_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  auto err = module->load();
  if (err != Error::Ok) {
    fprintf(stderr, "Load failed: 0x%x\n", static_cast<int>(err));
    return 1;
  }
  printf("Model loaded.\n");

  int vocab_size = static_cast<int>(strlen(VOCAB));
  float temperature = static_cast<float>(FLAGS_temperature);

  // Build char-to-index map
  int char_to_idx[256];
  memset(char_to_idx, 0, sizeof(char_to_idx));
  for (int i = 0; i < vocab_size; i++) {
    char_to_idx[static_cast<unsigned char>(VOCAB[i])] = i;
  }

  // Encode prompt
  std::vector<int64_t> prompt_ids;
  for (char c : FLAGS_prompt) {
    prompt_ids.push_back(char_to_idx[static_cast<unsigned char>(c)]);
  }

  printf("Prompt: \"%s\" (%zu tokens)\n\n", FLAGS_prompt.c_str(),
         prompt_ids.size());

  // Helper: read logits and sample next token
  auto sample_token = [&](const auto& logits_tensor) -> int64_t {
    int out_vocab = static_cast<int>(logits_tensor.size(2));
    const void* data = logits_tensor.const_data_ptr();
    auto dtype = logits_tensor.scalar_type();

    // Extract float logits
    std::vector<float> logits(out_vocab);
    for (int v = 0; v < out_vocab; v++) {
      if (dtype == exec_aten::ScalarType::Float) {
        logits[v] = static_cast<const float*>(data)[v];
      } else if (dtype == exec_aten::ScalarType::BFloat16) {
        uint32_t bits = static_cast<uint32_t>(
            static_cast<const uint16_t*>(data)[v]) << 16;
        std::memcpy(&logits[v], &bits, sizeof(float));
      }
    }

    if (temperature <= 0.01f) {
      // Greedy
      return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }

    // Temperature + softmax + multinomial
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0;
    for (int v = 0; v < out_vocab; v++) {
      logits[v] = expf((logits[v] - max_val) / temperature);
      sum += logits[v];
    }
    float r = static_cast<float>(rand()) / RAND_MAX * sum;
    float cumsum = 0;
    for (int v = 0; v < out_vocab; v++) {
      cumsum += logits[v];
      if (cumsum >= r) return v;
    }
    return out_vocab - 1;
  };

  // Helper: run one forward step
  auto forward_step = [&](int64_t token, int64_t pos) ->
      ::executorch::runtime::Result<std::vector<EValue>> {
    auto tok = from_blob(&token, {1, 1}, exec_aten::ScalarType::Long);
    auto p = from_blob(&pos, {1}, exec_aten::ScalarType::Long);
    std::vector<EValue> inputs;
    inputs.push_back(*tok);
    inputs.push_back(*p);
    return module->execute("forward", inputs);
  };

  srand(42);
  int64_t pos = 0;

  // Prefill: feed prompt tokens
  for (size_t i = 0; i < prompt_ids.size(); i++) {
    auto result = forward_step(prompt_ids[i], pos++);
    if (!result.ok()) {
      fprintf(stderr, "Prefill failed at %zu: 0x%x\n",
              i, static_cast<int>(result.error()));
      return 1;
    }
    // Print prompt chars
    if (prompt_ids[i] >= 0 && prompt_ids[i] < vocab_size) {
      putchar(VOCAB[prompt_ids[i]]);
    }

    // Sample from last prompt token
    if (i == prompt_ids.size() - 1) {
      int64_t next = sample_token(result.get()[0].toTensor());
      if (next >= 0 && next < vocab_size) putchar(VOCAB[next]);
      fflush(stdout);

      // Decode remaining tokens
      for (int step = 0; step < FLAGS_num_tokens - 1; step++) {
        if (pos >= 32) break; // block_size limit
        auto res = forward_step(next, pos++);
        if (!res.ok()) {
          fprintf(stderr, "\nDecode failed at step %d: 0x%x\n",
                  step, static_cast<int>(res.error()));
          return 1;
        }
        next = sample_token(res.get()[0].toTensor());
        if (next >= 0 && next < vocab_size) putchar(VOCAB[next]);
        fflush(stdout);
      }
    }
  }

  printf("\n\nDone.\n");
  return 0;
}
