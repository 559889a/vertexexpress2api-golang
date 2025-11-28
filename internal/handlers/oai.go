package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"vertex2api-golang/internal/keys"
	"vertex2api-golang/internal/models"
)

const (
	// ThinkingTagMarker is the tag used to mark thinking/reasoning content
	ThinkingTagMarker = "vertex_think_tag"
)

var (
	keyManager *keys.KeyManager
	httpClient *http.Client

	// Safety settings to disable content filtering
	safetySettings = []map[string]string{
		{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
		{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
		{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
		{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
		{"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
	}
)

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

	// Update model in request body with google/ prefix and add thinking config
	var reqMap map[string]interface{}
	json.Unmarshal(body, &reqMap)
	reqMap["model"] = vertexModelID

	// Add google extra_body for thinking chain support
	googleConfig := map[string]interface{}{
		"safety_settings":    safetySettings,
		"thought_tag_marker": ThinkingTagMarker,
		"thinking_config": map[string]interface{}{
			"include_thoughts": true,
		},
	}
	reqMap["google"] = googleConfig

	body, _ = json.Marshal(reqMap)

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
	var respMap map[string]interface{}
	if err := json.Unmarshal(respBody, &respMap); err != nil {
		return respBody
	}

	choices, ok := respMap["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return respBody
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return respBody
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return respBody
	}

	// Remove extra_content if present
	delete(message, "extra_content")

	content, ok := message["content"].(string)
	if !ok || content == "" {
		return respBody
	}

	// Extract reasoning from thinking tags
	reasoning, actualContent := extractReasoningByTags(content, ThinkingTagMarker)
	message["content"] = actualContent
	if reasoning != "" {
		message["reasoning_content"] = reasoning
		log.Printf("Extracted reasoning: %d chars, content: %d chars", len(reasoning), len(actualContent))
	}

	result, err := json.Marshal(respMap)
	if err != nil {
		return respBody
	}
	return result
}

// extractReasoningByTags extracts content between thinking tags
func extractReasoningByTags(content, tagName string) (reasoning, actualContent string) {
	openTag := "<" + tagName + ">"
	closeTag := "</" + tagName + ">"

	// Find all reasoning content within tags
	var reasoningParts []string
	remaining := content

	for {
		startIdx := strings.Index(remaining, openTag)
		if startIdx == -1 {
			break
		}

		endIdx := strings.Index(remaining[startIdx:], closeTag)
		if endIdx == -1 {
			break
		}
		endIdx += startIdx

		// Extract reasoning content
		reasoningContent := remaining[startIdx+len(openTag) : endIdx]
		reasoningParts = append(reasoningParts, reasoningContent)

		// Remove the tag and its content from remaining
		remaining = remaining[:startIdx] + remaining[endIdx+len(closeTag):]
	}

	reasoning = strings.Join(reasoningParts, "\n")
	actualContent = strings.TrimSpace(remaining)
	return
}

// StreamingReasoningProcessor handles extraction of reasoning from streaming chunks
type StreamingReasoningProcessor struct {
	tagName       string
	openTag       string
	closeTag      string
	insideTag     bool
	tagBuffer     string
	reasoningBuf  strings.Builder
}

// NewStreamingReasoningProcessor creates a new processor
func NewStreamingReasoningProcessor(tagName string) *StreamingReasoningProcessor {
	return &StreamingReasoningProcessor{
		tagName:  tagName,
		openTag:  "<" + tagName + ">",
		closeTag: "</" + tagName + ">",
	}
}

// ProcessChunk processes a content chunk and returns (processedContent, reasoningContent)
func (p *StreamingReasoningProcessor) ProcessChunk(content string) (processedContent, reasoningContent string) {
	// Add content to buffer for processing
	p.tagBuffer += content

	var contentParts []string
	var reasoningParts []string

	for len(p.tagBuffer) > 0 {
		if p.insideTag {
			// Look for closing tag
			closeIdx := strings.Index(p.tagBuffer, p.closeTag)
			if closeIdx == -1 {
				// No closing tag yet, might be partial - keep in buffer
				// But check if we have enough to determine it's definitely not the close tag
				if len(p.tagBuffer) > len(p.closeTag) {
					// Output up to last len(closeTag) chars as reasoning
					reasoningParts = append(reasoningParts, p.tagBuffer[:len(p.tagBuffer)-len(p.closeTag)])
					p.tagBuffer = p.tagBuffer[len(p.tagBuffer)-len(p.closeTag):]
				}
				break
			}
			// Found closing tag
			reasoningParts = append(reasoningParts, p.tagBuffer[:closeIdx])
			p.tagBuffer = p.tagBuffer[closeIdx+len(p.closeTag):]
			p.insideTag = false
		} else {
			// Look for opening tag
			openIdx := strings.Index(p.tagBuffer, p.openTag)
			if openIdx == -1 {
				// No opening tag - check for partial match at end
				for i := 1; i < len(p.openTag) && i <= len(p.tagBuffer); i++ {
					if strings.HasPrefix(p.openTag, p.tagBuffer[len(p.tagBuffer)-i:]) {
						// Potential partial tag at end
						contentParts = append(contentParts, p.tagBuffer[:len(p.tagBuffer)-i])
						p.tagBuffer = p.tagBuffer[len(p.tagBuffer)-i:]
						goto done
					}
				}
				// No partial match, output all as content
				contentParts = append(contentParts, p.tagBuffer)
				p.tagBuffer = ""
				break
			}
			// Found opening tag
			if openIdx > 0 {
				contentParts = append(contentParts, p.tagBuffer[:openIdx])
			}
			p.tagBuffer = p.tagBuffer[openIdx+len(p.openTag):]
			p.insideTag = true
		}
	}
done:
	processedContent = strings.Join(contentParts, "")
	reasoningContent = strings.Join(reasoningParts, "")
	return
}

// FlushRemaining returns any remaining buffered content
func (p *StreamingReasoningProcessor) FlushRemaining() (content, reasoning string) {
	if p.insideTag {
		// Unclosed tag - treat remaining as reasoning
		return "", p.tagBuffer
	}
	return p.tagBuffer, ""
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

			// Parse the chunk
			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
				// Can't parse, forward as-is
				sendSSE(jsonStr)
				continue
			}

			// Extract content from delta
			choices, ok := chunk["choices"].([]interface{})
			if !ok || len(choices) == 0 {
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
				continue
			}

			choice, ok := choices[0].(map[string]interface{})
			if !ok {
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
				continue
			}

			delta, ok := choice["delta"].(map[string]interface{})
			if !ok {
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
				continue
			}

			// Remove extra_content if present
			delete(delta, "extra_content")

			content, hasContent := delta["content"].(string)
			if !hasContent || content == "" {
				// No content to process, forward as-is (might have finish_reason)
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
				continue
			}

			// Process content for reasoning tags
			processedContent, reasoningContent := processor.ProcessChunk(content)

			// Send reasoning chunk if any
			if reasoningContent != "" {
				reasoningDelta := map[string]interface{}{
					"reasoning_content": reasoningContent,
				}
				reasoningChunk := map[string]interface{}{
					"id":      chunk["id"],
					"object":  chunk["object"],
					"created": chunk["created"],
					"model":   chunk["model"],
					"choices": []interface{}{
						map[string]interface{}{
							"index":         0,
							"delta":         reasoningDelta,
							"finish_reason": nil,
						},
					},
				}
				reasoningJSON, _ := json.Marshal(reasoningChunk)
				sendSSE(string(reasoningJSON))
			}

			// Send content chunk if any
			if processedContent != "" {
				delta["content"] = processedContent
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
			} else if choice["finish_reason"] != nil {
				// Has finish_reason but no content - forward the chunk
				delete(delta, "content")
				outputChunk, _ := json.Marshal(chunk)
				sendSSE(string(outputChunk))
			}
		}
	}

	// Flush remaining buffer
	remainingContent, remainingReasoning := processor.FlushRemaining()
	if remainingReasoning != "" {
		flushChunk := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-flush-%d", time.Now().Unix()),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   "unknown",
			"choices": []interface{}{
				map[string]interface{}{
					"index":         0,
					"delta":         map[string]interface{}{"reasoning_content": remainingReasoning},
					"finish_reason": nil,
				},
			},
		}
		flushJSON, _ := json.Marshal(flushChunk)
		sendSSE(string(flushJSON))
	}
	if remainingContent != "" {
		flushChunk := map[string]interface{}{
			"id":      fmt.Sprintf("chatcmpl-flush-%d", time.Now().Unix()),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   "unknown",
			"choices": []interface{}{
				map[string]interface{}{
					"index":         0,
					"delta":         map[string]interface{}{"content": remainingContent},
					"finish_reason": nil,
				},
			},
		}
		flushJSON, _ := json.Marshal(flushChunk)
		sendSSE(string(flushJSON))
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

	resp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errType,
			"code":    statusCode,
		},
	}
	json.NewEncoder(w).Encode(resp)
}
