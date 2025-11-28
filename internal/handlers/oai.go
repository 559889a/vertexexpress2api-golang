package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"

	"vertex2api-golang/internal/keys"
	"vertex2api-golang/internal/models"
	"vertex2api-golang/internal/vertex"
)

const (
	// ThinkingTagMarker is the tag used to mark thinking/reasoning content
	ThinkingTagMarker = "vertex_think_tag"
)

var (
	keyManager *keys.KeyManager
	httpClient *http.Client

	// reasoningTagPattern matches the thinking tag and its content
	reasoningTagPattern = regexp.MustCompile(`<` + ThinkingTagMarker + `>([\s\S]*?)</` + ThinkingTagMarker + `>`)

	// safetySettings disables content filtering
	safetySettings = []vertex.SafetySetting{
		{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_NONE"},
		{Category: "HARM_CATEGORY_CIVIC_INTEGRITY", Threshold: "BLOCK_NONE"},
	}
)

// OpenAI-compatible request/response types for the proxy endpoint

// chatRequest is the minimal request structure for parsing incoming requests
type chatRequest struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

// proxyRequest is the full request structure sent to Vertex AI OpenAI endpoint
type proxyRequest struct {
	Model  string       `json:"model"`
	Google googleConfig `json:"google"`
}

// googleConfig contains Vertex AI specific configuration
type googleConfig struct {
	SafetySettings   []vertex.SafetySetting `json:"safety_settings"`
	ThoughtTagMarker string                 `json:"thought_tag_marker"`
	ThinkingConfig   thinkingConfig         `json:"thinking_config"`
}

type thinkingConfig struct {
	IncludeThoughts bool `json:"include_thoughts"`
}

// streamChunk represents a parsed SSE chunk for streaming responses
type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

type streamChoice struct {
	Index        int         `json:"index"`
	Delta        streamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type streamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// nonStreamResponse represents the non-streaming API response
type nonStreamResponse struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []responseChoice  `json:"choices"`
	Usage   *responseUsage    `json:"usage,omitempty"`
}

type responseChoice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type responseMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type responseUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// errorResponse represents an OpenAI-compatible error response
type errorResponse struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    int    `json:"code"`
}

// InitClient initializes the vertex client (call after config is loaded)
func InitClient() {
	keyManager = keys.GetManager()
	httpClient = keyManager.GetHTTPClient()
}

// ModelsHandler handles /v1/models endpoint
func ModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Method not allowed")
		return
	}

	resp := models.GetModelsResponse()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ChatCompletionsHandler handles /v1/chat/completions endpoint
func ChatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Method not allowed")
		return
	}

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, http.StatusBadRequest, "invalid_request", "Failed to read request body")
		return
	}
	defer r.Body.Close()

	// Parse to get model and stream flag
	var req struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		sendError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON: "+err.Error())
		return
	}

	if req.Model == "" {
		sendError(w, http.StatusBadRequest, "invalid_request", "Model is required")
		return
	}

	// Resolve model alias
	actualModel, _ := models.ResolveModel(req.Model)

	// OpenAI-compatible endpoint requires "google/" prefix
	vertexModelID := "google/" + actualModel

	log.Printf("ChatCompletions: model=%s (actual=%s, vertex=%s), stream=%v", req.Model, actualModel, vertexModelID, req.Stream)

	// Build the request with google config for thinking chain support
	// We merge the original request with our additions using a two-pass approach
	var rawReq map[string]json.RawMessage
	if err := json.Unmarshal(body, &rawReq); err != nil {
		sendError(w, http.StatusBadRequest, "invalid_request", "Invalid JSON: "+err.Error())
		return
	}

	// Set the model with google/ prefix
	modelBytes, err := json.Marshal(vertexModelID)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "server_error", "Failed to encode model")
		return
	}
	rawReq["model"] = modelBytes

	// Add google config for thinking chain support
	gConfig := googleConfig{
		SafetySettings:   safetySettings,
		ThoughtTagMarker: ThinkingTagMarker,
		ThinkingConfig:   thinkingConfig{IncludeThoughts: true},
	}
	googleBytes, err := json.Marshal(gConfig)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "server_error", "Failed to encode google config")
		return
	}
	rawReq["google"] = googleBytes

	body, err = json.Marshal(rawReq)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "server_error", "Failed to encode request")
		return
	}

	// Forward to Vertex AI OpenAI-compatible endpoint
	ctx := r.Context()
	retryConfig := keys.GetRetryConfig()
	var lastErr error
	keyIndex := -1

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		var auth *keys.AuthInfo
		var err error

		if keyIndex < 0 {
			auth, err = keyManager.PickAuth(ctx)
		} else {
			auth, err = keyManager.PickAuthAtIndex(ctx, keyIndex)
		}

		if err != nil {
			sendError(w, http.StatusInternalServerError, "server_error", "Failed to get auth: "+err.Error())
			return
		}

		// Build Vertex AI OpenAI-compatible endpoint URL
		// Format: https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions?key={key}
		url := fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/endpoints/openapi/chat/completions?key=%s",
			auth.ProjectID,
			auth.Location,
			auth.APIKey,
		)

		startTime := time.Now()

		if req.Stream {
			err = handleStreamingProxy(w, url, body)
		} else {
			err = handleNonStreamingProxy(w, url, body)
		}

		latency := time.Since(startTime)

		if err == nil {
			log.Printf("ChatCompletions success: model=%s, key_index=%d, latency=%v", actualModel, auth.KeyIndex, latency)
			return
		}

		lastErr = err
		log.Printf("ChatCompletions attempt %d failed: model=%s, key_index=%d, error=%v", attempt+1, actualModel, auth.KeyIndex, err)

		// Switch to next key for retry
		if retryConfig.SwitchKey && keyManager.KeyCount() > 1 {
			keyIndex = keyManager.NextKeyIndex(auth.KeyIndex)
		}

		if attempt < retryConfig.MaxRetries {
			time.Sleep(time.Duration(retryConfig.IntervalMS) * time.Millisecond)
		}
	}

	sendError(w, http.StatusInternalServerError, "server_error", "All retries exhausted: "+lastErr.Error())
}

func handleNonStreamingProxy(w http.ResponseWriter, url string, body []byte) error {
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	// Process response to extract reasoning content
	respBody = processNonStreamingResponse(respBody)

	// Forward response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)

	return nil
}

// processNonStreamingResponse extracts reasoning from thinking tags and adds reasoning_content field
func processNonStreamingResponse(respBody []byte) []byte {
	var resp nonStreamResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return respBody
	}

	if len(resp.Choices) == 0 {
		return respBody
	}

	// Process the first choice's message content
	content := resp.Choices[0].Message.Content
	if content == "" {
		return respBody
	}

	// Extract reasoning from thinking tags using regexp
	reasoning, actualContent := extractReasoningByTags(content)
	resp.Choices[0].Message.Content = actualContent
	if reasoning != "" {
		resp.Choices[0].Message.ReasoningContent = reasoning
		log.Printf("Extracted reasoning: %d chars, content: %d chars", len(reasoning), len(actualContent))
	}

	result, err := json.Marshal(resp)
	if err != nil {
		return respBody
	}
	return result
}

// extractReasoningByTags extracts content between thinking tags using regexp
func extractReasoningByTags(content string) (reasoning, actualContent string) {
	matches := reasoningTagPattern.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return "", content
	}

	// Collect all reasoning parts
	reasoningParts := make([]string, 0, len(matches))
	for _, match := range matches {
		reasoningParts = append(reasoningParts, match[1])
	}

	// Remove all tags from content
	actualContent = strings.TrimSpace(reasoningTagPattern.ReplaceAllString(content, ""))
	reasoning = strings.Join(reasoningParts, "\n")
	return
}

// StreamingReasoningProcessor handles extraction of reasoning from streaming chunks
// using a simple state machine approach
type StreamingReasoningProcessor struct {
	openTag   string
	closeTag  string
	inTag     bool
	buffer    strings.Builder
	content   strings.Builder
	reasoning strings.Builder
}

// NewStreamingReasoningProcessor creates a new processor
func NewStreamingReasoningProcessor(tagName string) *StreamingReasoningProcessor {
	return &StreamingReasoningProcessor{
		openTag:  "<" + tagName + ">",
		closeTag: "</" + tagName + ">",
	}
}

// ProcessChunk processes a content chunk and returns (processedContent, reasoningContent)
func (p *StreamingReasoningProcessor) ProcessChunk(chunk string) (processedContent, reasoningContent string) {
	p.buffer.WriteString(chunk)
	buf := p.buffer.String()

	for {
		if p.inTag {
			idx := strings.Index(buf, p.closeTag)
			if idx < 0 {
				// Keep buffer minus the potential partial close tag
				keep := max(0, len(buf)-len(p.closeTag)+1)
				p.reasoning.WriteString(buf[:keep])
				p.buffer.Reset()
				p.buffer.WriteString(buf[keep:])
				break
			}
			p.reasoning.WriteString(buf[:idx])
			buf = buf[idx+len(p.closeTag):]
			p.inTag = false
		} else {
			idx := strings.Index(buf, p.openTag)
			if idx < 0 {
				// Check for partial open tag at the end
				partialIdx := p.findPartialTagStart(buf)
				if partialIdx >= 0 {
					p.content.WriteString(buf[:partialIdx])
					p.buffer.Reset()
					p.buffer.WriteString(buf[partialIdx:])
				} else {
					p.content.WriteString(buf)
					p.buffer.Reset()
				}
				break
			}
			p.content.WriteString(buf[:idx])
			buf = buf[idx+len(p.openTag):]
			p.inTag = true
		}
	}

	// Return accumulated content and reasoning, then reset accumulators
	processedContent = p.content.String()
	reasoningContent = p.reasoning.String()
	p.content.Reset()
	p.reasoning.Reset()
	return
}

// findPartialTagStart finds where a potential partial open tag starts at the end of buf
func (p *StreamingReasoningProcessor) findPartialTagStart(buf string) int {
	for i := 1; i < len(p.openTag) && i <= len(buf); i++ {
		if buf[len(buf)-i:] == p.openTag[:i] {
			return len(buf) - i
		}
	}
	return -1
}

// FlushRemaining returns any remaining buffered content
func (p *StreamingReasoningProcessor) FlushRemaining() (content, reasoning string) {
	buf := p.buffer.String()
	if p.inTag {
		return "", buf
	}
	return buf, ""
}

func handleStreamingProxy(w http.ResponseWriter, url string, body []byte) error {
	log.Printf("handleStreamingProxy: starting request")

	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("handleStreamingProxy: response status=%d", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		// Read error response body for logging; ignore read errors on error path
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("handleStreamingProxy: error response: %s", string(respBody))
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("handleStreamingProxy: flusher not available")
		return fmt.Errorf("streaming not supported")
	}

	log.Printf("handleStreamingProxy: flusher available, starting stream")

	// Create reasoning processor
	processor := NewStreamingReasoningProcessor(ThinkingTagMarker)

	// Helper to send SSE message with proper format (data: json\n\n)
	sendSSE := func(data string) {
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Stream response
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		lineCount++

		// Skip empty lines
		if line == "" {
			continue
		}

		// Process data lines for reasoning extraction
		if strings.HasPrefix(line, "data: ") {
			jsonStr := strings.TrimPrefix(line, "data: ")
			if jsonStr == "[DONE]" {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				continue
			}

			// Parse the chunk using typed struct
			var chunk streamChunk
			if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
				// Can't parse, forward as-is
				sendSSE(jsonStr)
				continue
			}

			// Check if we have content to process
			if len(chunk.Choices) == 0 {
				sendSSE(jsonStr)
				continue
			}

			content := chunk.Choices[0].Delta.Content
			if content == "" {
				// No content to process, forward as-is (might have finish_reason)
				sendSSE(jsonStr)
				continue
			}

			// Process content for reasoning tags
			processedContent, reasoningContent := processor.ProcessChunk(content)

			// Send reasoning chunk if any
			if reasoningContent != "" {
				reasoningChunk := streamChunk{
					ID:      chunk.ID,
					Object:  chunk.Object,
					Created: chunk.Created,
					Model:   chunk.Model,
					Choices: []streamChoice{{
						Index: 0,
						Delta: streamDelta{ReasoningContent: reasoningContent},
					}},
				}
				if reasoningJSON, err := json.Marshal(reasoningChunk); err == nil {
					sendSSE(string(reasoningJSON))
				}
			}

			// Send content chunk if any
			if processedContent != "" {
				chunk.Choices[0].Delta.Content = processedContent
				if outputChunk, err := json.Marshal(chunk); err == nil {
					sendSSE(string(outputChunk))
				}
			} else if chunk.Choices[0].FinishReason != nil {
				// Has finish_reason but no content - forward the chunk without content
				chunk.Choices[0].Delta.Content = ""
				if outputChunk, err := json.Marshal(chunk); err == nil {
					sendSSE(string(outputChunk))
				}
			}
		}
	}

	// Flush remaining buffer
	remainingContent, remainingReasoning := processor.FlushRemaining()
	now := time.Now().Unix()
	if remainingReasoning != "" {
		flushChunk := streamChunk{
			ID:      fmt.Sprintf("chatcmpl-flush-%d", now),
			Object:  "chat.completion.chunk",
			Created: now,
			Model:   "unknown",
			Choices: []streamChoice{{
				Index: 0,
				Delta: streamDelta{ReasoningContent: remainingReasoning},
			}},
		}
		if flushJSON, err := json.Marshal(flushChunk); err == nil {
			sendSSE(string(flushJSON))
		}
	}
	if remainingContent != "" {
		flushChunk := streamChunk{
			ID:      fmt.Sprintf("chatcmpl-flush-%d", now),
			Object:  "chat.completion.chunk",
			Created: now,
			Model:   "unknown",
			Choices: []streamChoice{{
				Index: 0,
				Delta: streamDelta{Content: remainingContent},
			}},
		}
		if flushJSON, err := json.Marshal(flushChunk); err == nil {
			sendSSE(string(flushJSON))
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("handleStreamingProxy: scanner error: %v", err)
		return fmt.Errorf("stream read error: %w", err)
	}

	log.Printf("handleStreamingProxy: stream completed, lines=%d", lineCount)
	return nil
}

func sendError(w http.ResponseWriter, statusCode int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	resp := errorResponse{
		Error: errorDetail{
			Message: message,
			Type:    errType,
			Code:    statusCode,
		},
	}
	json.NewEncoder(w).Encode(resp)
}
