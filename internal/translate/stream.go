package translate

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"vertex2api-golang/internal/vertex"
)

// StreamState tracks state for streaming response parsing
type StreamState struct {
	inThinking     bool
	thinkingBuffer strings.Builder
	contentBuffer  strings.Builder
}

// NewStreamState creates a new stream state
func NewStreamState() *StreamState {
	return &StreamState{}
}

// ProcessChunk processes a streaming chunk and extracts content/reasoning
func (s *StreamState) ProcessChunk(chunk *vertex.GeminiResponse) (content string, reasoning string, toolCalls []ToolCall, finishReason string) {
	if chunk == nil || len(chunk.Candidates) == 0 {
		return
	}

	candidate := chunk.Candidates[0]
	finishReason = mapFinishReason(candidate.FinishReason)

	if candidate.Content == nil {
		return
	}

	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			c, r := s.processText(part.Text)
			content += c
			reasoning += r
		}

		if part.FunctionCall != nil {
			args, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, ToolCall{
				ID:   generateToolCallID(),
				Type: "function",
				Function: FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(args),
				},
			})
		}
	}

	return
}

// processText handles thinking tag parsing with state machine
func (s *StreamState) processText(text string) (content string, reasoning string) {
	// Pattern for thinking tags
	openTag := "<vertex_think_tag>"
	closeTag := "</vertex_think_tag>"

	remaining := text

	for len(remaining) > 0 {
		if s.inThinking {
			// Looking for close tag
			closeIdx := strings.Index(remaining, closeTag)
			if closeIdx >= 0 {
				// Found close tag
				s.thinkingBuffer.WriteString(remaining[:closeIdx])
				reasoning = s.thinkingBuffer.String()
				s.thinkingBuffer.Reset()
				s.inThinking = false
				remaining = remaining[closeIdx+len(closeTag):]
			} else {
				// No close tag yet, buffer everything
				s.thinkingBuffer.WriteString(remaining)
				remaining = ""
			}
		} else {
			// Looking for open tag
			openIdx := strings.Index(remaining, openTag)
			if openIdx >= 0 {
				// Found open tag
				content += remaining[:openIdx]
				s.inThinking = true
				remaining = remaining[openIdx+len(openTag):]
			} else {
				// Check for partial tag at end
				partialIdx := findPartialTag(remaining, openTag)
				if partialIdx >= 0 {
					content += remaining[:partialIdx]
					s.contentBuffer.WriteString(remaining[partialIdx:])
					remaining = ""
				} else {
					content += remaining
					remaining = ""
				}
			}
		}
	}

	return
}

// findPartialTag finds index where a partial tag match might start
func findPartialTag(text string, tag string) int {
	for i := 1; i < len(tag) && i <= len(text); i++ {
		suffix := text[len(text)-i:]
		prefix := tag[:i]
		if suffix == prefix {
			return len(text) - i
		}
	}
	return -1
}

// StreamChunkResponse represents a streaming chunk response
type StreamChunkResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	Usage             *Usage   `json:"usage,omitempty"`
}

// SSEWriter handles SSE output
type SSEWriter struct {
	w         http.ResponseWriter
	flusher   http.Flusher
	requestID string
	model     string
	created   int64
}

// NewSSEWriter creates a new SSE writer
func NewSSEWriter(w http.ResponseWriter, requestID, model string) *SSEWriter {
	flusher, _ := w.(http.Flusher)

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")
	w.Header().Set("X-Accel-Buffering", "no")

	return &SSEWriter{
		w:         w,
		flusher:   flusher,
		requestID: requestID,
		model:     model,
		created:   time.Now().Unix(),
	}
}

// WriteChunk writes a streaming chunk
func (s *SSEWriter) WriteChunk(content, reasoning string, toolCalls []ToolCall, finishReason string, isFirst bool, usage *Usage) error {
	chunk := StreamChunkResponse{
		ID:      s.requestID,
		Object:  "chat.completion.chunk",
		Created: s.created,
		Model:   s.model,
		Choices: []Choice{{
			Index: 0,
			Delta: &ResponseMsg{},
		}},
	}

	// Set role on first chunk
	if isFirst {
		chunk.Choices[0].Delta.Role = "assistant"
	}

	// Set content
	if content != "" {
		chunk.Choices[0].Delta.Content = content
	}

	// Set reasoning
	if reasoning != "" {
		chunk.Choices[0].Delta.ReasoningContent = reasoning
	}

	// Set tool calls
	if len(toolCalls) > 0 {
		chunk.Choices[0].Delta.ToolCalls = toolCalls
	}

	// Set finish reason
	if finishReason != "" {
		chunk.Choices[0].FinishReason = finishReason
	}

	// Set usage on final chunk
	if usage != nil {
		chunk.Usage = usage
	}

	return s.writeSSE(chunk)
}

// WriteDone writes the final [DONE] message
func (s *SSEWriter) WriteDone() error {
	_, err := fmt.Fprintf(s.w, "data: [DONE]\n\n")
	if err != nil {
		return err
	}
	if s.flusher != nil {
		s.flusher.Flush()
	}
	return nil
}

// WriteError writes an error as SSE
func (s *SSEWriter) WriteError(errMsg string) error {
	errResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": errMsg,
			"type":    "server_error",
		},
	}
	return s.writeSSE(errResp)
}

func (s *SSEWriter) writeSSE(data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(s.w, "data: %s\n\n", jsonData)
	if err != nil {
		return err
	}

	if s.flusher != nil {
		s.flusher.Flush()
	}

	return nil
}

// ExtractThinkingFromText extracts thinking content using regex (for non-streaming)
func ExtractThinkingFromText(text string) (content string, reasoning string) {
	thinkPattern := regexp.MustCompile(`<vertex_think_tag>([\s\S]*?)</vertex_think_tag>`)
	matches := thinkPattern.FindAllStringSubmatch(text, -1)

	if len(matches) == 0 {
		return text, ""
	}

	var reasonings []string
	remaining := text

	for _, match := range matches {
		reasonings = append(reasonings, strings.TrimSpace(match[1]))
		remaining = strings.Replace(remaining, match[0], "", 1)
	}

	return strings.TrimSpace(remaining), strings.Join(reasonings, "\n")
}
