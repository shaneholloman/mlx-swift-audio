// Copyright © 2025 FunASR (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/modelscope/FunASR
// License: licenses/funasr.txt

import Foundation
import MLXLMCommon

/// Fun-ASR tokenizer wrapper for Qwen3
///
/// Uses a TokenizerLoader for Qwen3 tokenization.
class FunASRTokenizer {
  private let tokenizer: any MLXLMCommon.Tokenizer

  /// Token IDs that signal the end of generation.
  let eosTokenIds: Set<Int>

  // Configuration
  let config: FunASRConfig

  private init(tokenizer: any MLXLMCommon.Tokenizer, config: FunASRConfig) {
    self.tokenizer = tokenizer
    self.config = config

    // Build set of EOS token IDs for stopping generation.
    // The Fun-ASR speech markers (`<|startofspeech|>` / `<|endofspeech|>`) are
    // not added tokens — they decompose to multi-token BPE sequences and must
    // never be used as stop tokens.
    var eosIds = Set<Int>()
    let stopTokens = ["<|endoftext|>", "<|im_end|>", "</s>"]
    for token in stopTokens {
      let encoded = tokenizer.encode(text: token)
      // Only treat as a stop token if it tokenizes to a single ID, i.e. it is
      // an added token in the tokenizer's vocabulary.
      if encoded.count == 1, let id = encoded.first {
        eosIds.insert(id)
      }
    }
    if let tokenizerEos = tokenizer.eosTokenId {
      eosIds.insert(tokenizerEos)
    }
    eosTokenIds = eosIds
  }

  /// Load tokenizer from model directory
  ///
  /// - Parameters:
  ///   - modelDirectory: Path to model directory containing tokenizer files
  ///   - config: Fun-ASR configuration
  /// - Returns: Initialized tokenizer
  static func load(
    from directory: URL, config: FunASRConfig, using tokenizerLoader: any TokenizerLoader,
  ) async throws -> FunASRTokenizer {
    // Load tokenizer from the model directory
    let tokenizer = try await tokenizerLoader.load(from: directory)
    return FunASRTokenizer(tokenizer: tokenizer, config: config)
  }

  /// Encode text to token IDs
  ///
  /// - Parameter text: Text to encode
  /// - Returns: Array of token IDs
  func encode(_ text: String) -> [Int] {
    tokenizer.encode(text: text)
  }

  /// Decode token IDs to text
  ///
  /// - Parameter tokens: Array of token IDs
  /// - Returns: Decoded text
  func decode(_ tokens: [Int]) -> String {
    tokenizer.decode(tokenIds: tokens)
  }

  /// Check if a token ID is an EOS token
  ///
  /// - Parameter tokenId: Token ID to check
  /// - Returns: True if the token is an EOS token
  func isEosToken(_ tokenId: Int) -> Bool {
    eosTokenIds.contains(tokenId)
  }

  /// Build the prompt as the pair of token sequences that go before and after
  /// the audio embeddings.
  ///
  /// The PyTorch reference (`fun_asr_nano/model.py`) splits the source string
  /// on `<|startofspeech|>...<|endofspeech|>` and tokenizes the surrounding
  /// text. The marker substring is replaced by `fake_token_len` placeholder
  /// zeros that are later overwritten by audio embeddings. We avoid the
  /// placeholders entirely and splice at the embedding level: the LLM sees
  /// `embed(pre) | audio_embeddings | embed(post)`.
  ///
  /// The marker tokens themselves are never sent to the LLM — they decompose
  /// to BPE pieces in this tokenizer (`<`, `|`, `start`, ...), and shipping
  /// them would corrupt the prompt.
  ///
  /// - Parameters:
  ///   - task: Task type (transcribe or translate)
  ///   - language: Source language (or `.auto` for detection)
  ///   - targetLanguage: Target language for translation
  ///   - initialPrompt: Custom instructions to prepend to the user prompt
  /// - Returns: Token IDs for the text before the audio (`pre`) and after it
  ///   (`post`). Concatenating `embed(pre) + audio + embed(post)` gives the
  ///   full LLM input.
  func buildPromptParts(
    task: FunASRTask,
    language: FunASRLanguage = .auto,
    targetLanguage: FunASRLanguage = .english,
    initialPrompt: String? = nil,
  ) -> (pre: [Int], post: [Int]) {
    let systemPrompt = buildSystemPrompt(task: task)
    let userInstruction = buildUserInstruction(
      task: task,
      language: language,
      targetLanguage: targetLanguage,
      initialPrompt: initialPrompt,
    )

    let preText =
      "\(config.imStartToken)system\n\(systemPrompt)\(config.imEndToken)\n"
        + "\(config.imStartToken)user\n\(userInstruction)"
    let postText =
      "\(config.imEndToken)\n\(config.imStartToken)assistant\n"

    return (encode(preText), encode(postText))
  }

  /// System prompt matching the upstream Fun-ASR reference (`generate_chatml`).
  private func buildSystemPrompt(task _: FunASRTask) -> String {
    "You are a helpful assistant."
  }

  /// User instruction text up to (but not including) the speech markers.
  ///
  /// Matches upstream PyTorch's `get_prompt`:
  /// `语音转写：` for `.auto`, `语音转写成{language}：` otherwise.
  ///
  /// Upstream Fun-ASR has no separate translation template. The
  /// "transcribe to `{language}`" prompt with `language` set to the target
  /// effectively serves as translation when the source language differs
  /// from the target — the multilingual variant follows it; the
  /// transcription-only variant generally does not.
  private func buildUserInstruction(
    task: FunASRTask,
    language: FunASRLanguage,
    targetLanguage: FunASRLanguage,
    initialPrompt: String?,
  ) -> String {
    let promptLanguage: FunASRLanguage = switch task {
      case .transcribe: language
      case .translate: targetLanguage
    }
    var instruction = if let langName = promptLanguage.promptName {
      "语音转写成\(langName)："
    } else {
      "语音转写："
    }
    if let initialPrompt, !initialPrompt.isEmpty {
      instruction = "\(initialPrompt)\n\n\(instruction)"
    }
    return instruction
  }

  /// Clean output text by removing special tokens and artifacts.
  ///
  /// - Parameter text: Raw generated text
  /// - Returns: Cleaned text
  func cleanOutput(_ text: String) -> String {
    var cleaned = text

    // Remove thinking blocks
    let thinkPattern = #"<think>.*?</think>"#
    if let regex = try? NSRegularExpression(pattern: thinkPattern, options: .dotMatchesLineSeparators) {
      cleaned = regex.stringByReplacingMatches(
        in: cleaned,
        range: NSRange(cleaned.startIndex..., in: cleaned),
        withTemplate: "",
      )
    }

    // Strip any literal special token strings that may appear in the output.
    let specialTokens = [
      config.imStartToken,
      config.imEndToken,
      config.sosToken,
      config.eosToken,
      "<|endoftext|>",
    ]
    for token in specialTokens {
      cleaned = cleaned.replacingOccurrences(of: token, with: "")
    }

    return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
