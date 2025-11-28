package vertex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"vertex2api-golang/internal/keys"
)

// GeminiRequest represents a Gemini API request
type GeminiRequest struct {
	Contents          []Content          `json:"contents,omitempty"`
	SystemInstruction *Content           `json:"systemInstruction,omitempty"`
	GenerationConfig  *GenerationConfig  `json:"generationConfig,omitempty"`
	Tools             []Tool             `json:"tools,omitempty"`
	ToolConfig        *ToolConfig        `json:"toolConfig,omitempty"`
	SafetySettings    []SafetySetting    `json:"safetySettings,omitempty"`
}

// Content represents message content
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part represents a content part (text, image, function call, etc.)
type Part struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *InlineData       `json:"inlineData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

// InlineData represents inline binary data (images)
type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64 encoded
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args,omitempty"`
}

// FunctionResponse represents a function response
type FunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

// GenerationConfig contains generation parameters
type GenerationConfig struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	TopK             *int     `json:"topK,omitempty"`
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	CandidateCount   *int     `json:"candidateCount,omitempty"`
	ResponseMimeType string   `json:"responseMimeType,omitempty"`
	ThinkingConfig   *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

// ThinkingConfig for Gemini 3 thinking models
type ThinkingConfig struct {
	ThinkingBudget int `json:"thinkingBudget,omitempty"`
}

// Tool represents a function tool
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration declares a function
type FunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolConfig configures tool behavior
type ToolConfig struct {
	FunctionCallingConfig *FunctionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// FunctionCallingConfig configures function calling
type FunctionCallingConfig struct {
	Mode                 string   `json:"mode,omitempty"` // AUTO, ANY, NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// SafetySetting configures safety thresholds
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GeminiResponse represents a Gemini API response
type GeminiResponse struct {
	Candidates     []Candidate     `json:"candidates,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usageMetadata,omitempty"`
	PromptFeedback *PromptFeedback `json:"promptFeedback,omitempty"`
	ModelVersion   string          `json:"modelVersion,omitempty"`
}

// Candidate represents a response candidate
type Candidate struct {
	Content       *Content        `json:"content,omitempty"`
	FinishReason  string          `json:"finishReason,omitempty"`
	Index         int             `json:"index"`
	SafetyRatings []SafetyRating  `json:"safetyRatings,omitempty"`
}

// SafetyRating represents safety rating
type SafetyRating struct {
	Category    string  `json:"category"`
	Probability string  `json:"probability"`
	Score       float64 `json:"score,omitempty"`
}

// UsageMetadata contains token usage
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
	ThoughtsTokenCount   int `json:"thoughtsTokenCount,omitempty"`
}

// PromptFeedback contains prompt feedback
type PromptFeedback struct {
	BlockReason   string         `json:"blockReason,omitempty"`
	SafetyRatings []SafetyRating `json:"safetyRatings,omitempty"`
}

// Client wraps Vertex API calls
type Client struct {
	keyManager *keys.KeyManager
	httpClient *http.Client
}

// NewClient creates a new Vertex client
func NewClient() *Client {
	km := keys.GetManager()
	return &Client{
		keyManager: km,
		httpClient: km.GetHTTPClient(),
	}
}

// buildURL constructs the Vertex API URL
func (c *Client) buildURL(auth *keys.AuthInfo, model string, stream bool) string {
	action := "generateContent"
	if stream {
		action = "streamGenerateContent"
	}

	// URL format: https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model}:{action}
	return fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:%s?key=%s",
		auth.Location,
		auth.ProjectID,
		auth.Location,
		model,
		action,
		auth.APIKey,
	)
}

// GenerateContent calls the non-streaming API
func (c *Client) GenerateContent(ctx context.Context, model string, req *GeminiRequest) (*GeminiResponse, error) {
	retryConfig := keys.GetRetryConfig()
	var lastErr error
	var keyIndex int = -1

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		var auth *keys.AuthInfo
		var err error

		if keyIndex < 0 {
			auth, err = c.keyManager.PickAuth(ctx)
		} else {
			auth, err = c.keyManager.PickAuthAtIndex(ctx, keyIndex)
		}

		if err != nil {
			return nil, fmt.Errorf("failed to get auth: %w", err)
		}

		startTime := time.Now()
		resp, err := c.doRequest(ctx, auth, model, req, false)
		latency := time.Since(startTime)

		if err == nil {
			log.Printf("GenerateContent success: model=%s, key_index=%d, latency=%v", model, auth.KeyIndex, latency)
			return resp, nil
		}

		lastErr = err
		log.Printf("GenerateContent attempt %d failed: model=%s, key_index=%d, error=%v", attempt+1, model, auth.KeyIndex, err)

		// Switch to next key for retry
		if retryConfig.SwitchKey && c.keyManager.KeyCount() > 1 {
			keyIndex = c.keyManager.NextKeyIndex(auth.KeyIndex)
		}

		if attempt < retryConfig.MaxRetries {
			time.Sleep(time.Duration(retryConfig.IntervalMS) * time.Millisecond)
		}
	}

	return nil, fmt.Errorf("all retries exhausted: %w", lastErr)
}

// StreamGenerateContent calls the streaming API
func (c *Client) StreamGenerateContent(ctx context.Context, model string, req *GeminiRequest, handler StreamHandler) error {
	retryConfig := keys.GetRetryConfig()
	var lastErr error
	var keyIndex int = -1

	for attempt := 0; attempt <= retryConfig.MaxRetries; attempt++ {
		var auth *keys.AuthInfo
		var err error

		if keyIndex < 0 {
			auth, err = c.keyManager.PickAuth(ctx)
		} else {
			auth, err = c.keyManager.PickAuthAtIndex(ctx, keyIndex)
		}

		if err != nil {
			return fmt.Errorf("failed to get auth: %w", err)
		}

		startTime := time.Now()
		err = c.doStreamRequest(ctx, auth, model, req, handler)
		latency := time.Since(startTime)

		if err == nil {
			log.Printf("StreamGenerateContent success: model=%s, key_index=%d, latency=%v", model, auth.KeyIndex, latency)
			return nil
		}

		lastErr = err
		log.Printf("StreamGenerateContent attempt %d failed: model=%s, key_index=%d, error=%v", attempt+1, model, auth.KeyIndex, err)

		// Switch to next key for retry
		if retryConfig.SwitchKey && c.keyManager.KeyCount() > 1 {
			keyIndex = c.keyManager.NextKeyIndex(auth.KeyIndex)
		}

		if attempt < retryConfig.MaxRetries {
			time.Sleep(time.Duration(retryConfig.IntervalMS) * time.Millisecond)
		}
	}

	return fmt.Errorf("all retries exhausted: %w", lastErr)
}

func (c *Client) doRequest(ctx context.Context, auth *keys.AuthInfo, model string, geminiReq *GeminiRequest, stream bool) (*GeminiResponse, error) {
	url := c.buildURL(auth, model, stream)

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var geminiResp GeminiResponse
	if err := json.Unmarshal(respBody, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &geminiResp, nil
}

// StreamHandler handles streaming chunks
type StreamHandler func(chunk *GeminiResponse) error

func (c *Client) doStreamRequest(ctx context.Context, auth *keys.AuthInfo, model string, geminiReq *GeminiRequest, handler StreamHandler) error {
	url := c.buildURL(auth, model, true) + "&alt=sse"

	body, err := json.Marshal(geminiReq)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read error response body for logging; ignore read errors on error path
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	// Parse SSE stream
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk GeminiResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("Failed to parse SSE chunk: %v", err)
			continue
		}

		if err := handler(&chunk); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("stream read error: %w", err)
	}

	return nil
}

// ForwardRaw forwards a raw request to Vertex API (for Gemini native endpoints)
func (c *Client) ForwardRaw(ctx context.Context, model, action string, reqBody []byte) (*http.Response, error) {
	auth, err := c.keyManager.PickAuth(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get auth: %w", err)
	}

	url := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/%s/locations/%s/publishers/google/models/%s:%s?key=%s",
		auth.Location,
		auth.ProjectID,
		auth.Location,
		model,
		action,
		auth.APIKey,
	)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	return c.httpClient.Do(req)
}
