package translate

import (
	"encoding/base64"
	"encoding/json"
	"regexp"
	"strings"

	"vertex2api-golang/internal/models"
	"vertex2api-golang/internal/vertex"
)

// OpenAI request/response types

// ChatCompletionRequest represents OpenAI chat completion request
type ChatCompletionRequest struct {
	Model            string                 `json:"model"`
	Messages         []Message              `json:"messages"`
	Temperature      *float64               `json:"temperature,omitempty"`
	TopP             *float64               `json:"top_p,omitempty"`
	TopK             *int                   `json:"top_k,omitempty"`
	N                *int                   `json:"n,omitempty"`
	Stream           bool                   `json:"stream,omitempty"`
	Stop             interface{}            `json:"stop,omitempty"`
	MaxTokens        *int                   `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	PresencePenalty  *float64               `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64               `json:"frequency_penalty,omitempty"`
	LogitBias        map[string]float64     `json:"logit_bias,omitempty"`
	User             string                 `json:"user,omitempty"`
	Tools            []OpenAITool           `json:"tools,omitempty"`
	ToolChoice       interface{}            `json:"tool_choice,omitempty"`
	ResponseFormat   *ResponseFormat        `json:"response_format,omitempty"`
	Seed             *int                   `json:"seed,omitempty"`
	Logprobs         *bool                  `json:"logprobs,omitempty"`
	TopLogprobs      *int                   `json:"top_logprobs,omitempty"`
	// Extended fields
	SafetySettings   []vertex.SafetySetting `json:"safety_settings,omitempty"`
}

// Message represents an OpenAI message
type Message struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"` // string or []ContentPart
	Name       string      `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// ContentPart represents a content part in multimodal messages
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ToolCall represents an OpenAI tool call
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAITool represents an OpenAI tool
type OpenAITool struct {
	Type     string           `json:"type"`
	Function OpenAIFunction   `json:"function"`
}

// OpenAIFunction represents an OpenAI function definition
type OpenAIFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ResponseFormat specifies response format
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// ChatCompletionResponse represents OpenAI chat completion response
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a response choice
type Choice struct {
	Index        int            `json:"index"`
	Message      *ResponseMsg   `json:"message,omitempty"`
	Delta        *ResponseMsg   `json:"delta,omitempty"`
	FinishReason string         `json:"finish_reason,omitempty"`
	Logprobs     interface{}    `json:"logprobs,omitempty"`
}

// ResponseMsg represents response message
type ResponseMsg struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens            int `json:"prompt_tokens"`
	CompletionTokens        int `json:"completion_tokens"`
	TotalTokens             int `json:"total_tokens"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// CompletionTokensDetails contains detailed completion token info
type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ToGeminiRequest converts OpenAI request to Gemini request
func ToGeminiRequest(oaiReq *ChatCompletionRequest) (*vertex.GeminiRequest, string) {
	geminiReq := &vertex.GeminiRequest{}

	// Resolve model alias
	actualModel, alias := models.ResolveModel(oaiReq.Model)

	// Convert messages
	var systemParts []vertex.Part
	var contents []vertex.Content

	for _, msg := range oaiReq.Messages {
		switch msg.Role {
		case "system":
			// Collect system messages
			text := extractTextContent(msg.Content)
			if text != "" {
				systemParts = append(systemParts, vertex.Part{Text: text})
			}

		case "user":
			parts := convertContentToParts(msg.Content)
			if len(parts) > 0 {
				contents = append(contents, vertex.Content{
					Role:  "user",
					Parts: parts,
				})
			}

		case "assistant":
			content := vertex.Content{
				Role: "model",
			}

			// Handle tool calls
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					var args map[string]interface{}
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
						// If args can't be parsed, use empty map
						args = make(map[string]interface{})
					}
					content.Parts = append(content.Parts, vertex.Part{
						FunctionCall: &vertex.FunctionCall{
							Name: tc.Function.Name,
							Args: args,
						},
					})
				}
			} else {
				text := extractTextContent(msg.Content)
				if text != "" {
					content.Parts = append(content.Parts, vertex.Part{Text: text})
				}
			}

			if len(content.Parts) > 0 {
				contents = append(contents, content)
			}

		case "tool":
			// Tool response
			var respData map[string]interface{}
			text := extractTextContent(msg.Content)
			if err := json.Unmarshal([]byte(text), &respData); err != nil {
				respData = map[string]interface{}{"result": text}
			}

			contents = append(contents, vertex.Content{
				Role: "user",
				Parts: []vertex.Part{{
					FunctionResponse: &vertex.FunctionResponse{
						Name:     msg.Name,
						Response: respData,
					},
				}},
			})
		}
	}

	// Set system instruction
	if len(systemParts) > 0 {
		geminiReq.SystemInstruction = &vertex.Content{
			Parts: systemParts,
		}
	}

	geminiReq.Contents = contents

	// Convert generation config
	geminiReq.GenerationConfig = &vertex.GenerationConfig{}

	if oaiReq.Temperature != nil {
		geminiReq.GenerationConfig.Temperature = oaiReq.Temperature
	}
	if oaiReq.TopP != nil {
		geminiReq.GenerationConfig.TopP = oaiReq.TopP
	}
	if oaiReq.TopK != nil {
		geminiReq.GenerationConfig.TopK = oaiReq.TopK
	}

	// Max tokens
	maxTokens := oaiReq.MaxTokens
	if oaiReq.MaxCompletionTokens != nil {
		maxTokens = oaiReq.MaxCompletionTokens
	}
	if maxTokens != nil {
		geminiReq.GenerationConfig.MaxOutputTokens = maxTokens
	}

	// Stop sequences
	if oaiReq.Stop != nil {
		switch v := oaiReq.Stop.(type) {
		case string:
			geminiReq.GenerationConfig.StopSequences = []string{v}
		case []interface{}:
			for _, s := range v {
				if str, ok := s.(string); ok {
					geminiReq.GenerationConfig.StopSequences = append(geminiReq.GenerationConfig.StopSequences, str)
				}
			}
		}
	}

	// Candidate count
	if oaiReq.N != nil && *oaiReq.N > 1 {
		geminiReq.GenerationConfig.CandidateCount = oaiReq.N
	}

	// Response format
	if oaiReq.ResponseFormat != nil && oaiReq.ResponseFormat.Type == "json_object" {
		geminiReq.GenerationConfig.ResponseMimeType = "application/json"
	}

	// Thinking config for alias models
	if alias != nil && alias.ThinkingLevel != "" {
		budget := 1024 // low
		if alias.ThinkingLevel == "high" {
			budget = 8192
		}
		geminiReq.GenerationConfig.ThinkingConfig = &vertex.ThinkingConfig{
			ThinkingBudget: budget,
		}
	}

	// Convert tools
	if len(oaiReq.Tools) > 0 {
		var funcDecls []vertex.FunctionDeclaration
		for _, tool := range oaiReq.Tools {
			if tool.Type == "function" {
				funcDecls = append(funcDecls, vertex.FunctionDeclaration{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				})
			}
		}
		if len(funcDecls) > 0 {
			geminiReq.Tools = []vertex.Tool{{
				FunctionDeclarations: funcDecls,
			}}
		}
	}

	// Tool choice
	if oaiReq.ToolChoice != nil {
		geminiReq.ToolConfig = convertToolChoice(oaiReq.ToolChoice)
	}

	// Safety settings
	if len(oaiReq.SafetySettings) > 0 {
		geminiReq.SafetySettings = oaiReq.SafetySettings
	}

	return geminiReq, actualModel
}

// extractTextContent extracts text from OpenAI content field.
// Content can be either a string or an array of content parts.
func extractTextContent(content interface{}) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []interface{}:
		return extractTextFromParts(v)
	default:
		return ""
	}
}

// extractTextFromParts extracts text from content parts array
func extractTextFromParts(parts []interface{}) string {
	var texts []string
	for _, part := range parts {
		m, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		if m["type"] != "text" {
			continue
		}
		if text, ok := m["text"].(string); ok {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n")
}

// convertContentToParts converts OpenAI content to Gemini parts.
// Content can be either a string or an array of content parts.
func convertContentToParts(content interface{}) []vertex.Part {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []vertex.Part{{Text: v}}
	case []interface{}:
		return convertContentArrayToParts(v)
	default:
		return nil
	}
}

// convertContentArrayToParts handles array content conversion
func convertContentArrayToParts(items []interface{}) []vertex.Part {
	var parts []vertex.Part
	for _, item := range items {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		part := convertSingleContentPart(m)
		if part != nil {
			parts = append(parts, *part)
		}
	}
	return parts
}

// convertSingleContentPart converts a single content part map to a Gemini Part
func convertSingleContentPart(m map[string]interface{}) *vertex.Part {
	partType, _ := m["type"].(string)
	switch partType {
	case "text":
		text, ok := m["text"].(string)
		if !ok || text == "" {
			return nil
		}
		return &vertex.Part{Text: text}
	case "image_url":
		imgURL, ok := m["image_url"].(map[string]interface{})
		if !ok {
			return nil
		}
		url, ok := imgURL["url"].(string)
		if !ok {
			return nil
		}
		return parseImageURL(url)
	default:
		return nil
	}
}

// parseImageURL parses image URL (data URL or markdown base64)
func parseImageURL(url string) *vertex.Part {
	// Handle data URL: data:image/png;base64,xxxx
	if strings.HasPrefix(url, "data:") {
		parts := strings.SplitN(url, ",", 2)
		if len(parts) != 2 {
			return nil
		}

		// Extract mime type
		meta := parts[0] // data:image/png;base64
		mimeType := "image/png"
		if strings.Contains(meta, "image/jpeg") {
			mimeType = "image/jpeg"
		} else if strings.Contains(meta, "image/gif") {
			mimeType = "image/gif"
		} else if strings.Contains(meta, "image/webp") {
			mimeType = "image/webp"
		}

		return &vertex.Part{
			InlineData: &vertex.InlineData{
				MimeType: mimeType,
				Data:     parts[1],
			},
		}
	}

	// Handle markdown base64: ![](data:image/png;base64,xxxx)
	re := regexp.MustCompile(`!\[.*?\]\((data:[^)]+)\)`)
	if matches := re.FindStringSubmatch(url); len(matches) > 1 {
		return parseImageURL(matches[1])
	}

	// For regular URLs, we would need to fetch the image
	// For now, just skip external URLs
	return nil
}

func convertToolChoice(toolChoice interface{}) *vertex.ToolConfig {
	config := &vertex.ToolConfig{
		FunctionCallingConfig: &vertex.FunctionCallingConfig{},
	}

	switch v := toolChoice.(type) {
	case string:
		switch v {
		case "none":
			config.FunctionCallingConfig.Mode = "NONE"
		case "auto":
			config.FunctionCallingConfig.Mode = "AUTO"
		case "required":
			config.FunctionCallingConfig.Mode = "ANY"
		}

	case map[string]interface{}:
		if v["type"] == "function" {
			if fn, ok := v["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					config.FunctionCallingConfig.Mode = "ANY"
					config.FunctionCallingConfig.AllowedFunctionNames = []string{name}
				}
			}
		}
	}

	return config
}

// FromGeminiResponse converts Gemini response to OpenAI response
func FromGeminiResponse(geminiResp *vertex.GeminiResponse, model string, requestID string) *ChatCompletionResponse {
	resp := &ChatCompletionResponse{
		ID:      requestID,
		Object:  "chat.completion",
		Created: 0, // Will be set by caller
		Model:   model,
		Choices: make([]Choice, 0),
	}

	if geminiResp == nil {
		return resp
	}

	// Convert candidates to choices
	for i, candidate := range geminiResp.Candidates {
		choice := Choice{
			Index:        i,
			FinishReason: mapFinishReason(candidate.FinishReason),
			Message:      &ResponseMsg{Role: "assistant"},
		}

		if candidate.Content != nil {
			var textParts []string
			var reasoningParts []string

			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					// Check for thinking tags
					text, reasoning := extractThinking(part.Text)
					if text != "" {
						textParts = append(textParts, text)
					}
					if reasoning != "" {
						reasoningParts = append(reasoningParts, reasoning)
					}
				}

				if part.FunctionCall != nil {
					args, err := json.Marshal(part.FunctionCall.Args)
					if err != nil {
						args = []byte("{}")
					}
					choice.Message.ToolCalls = append(choice.Message.ToolCalls, ToolCall{
						ID:   generateToolCallID(),
						Type: "function",
						Function: FunctionCall{
							Name:      part.FunctionCall.Name,
							Arguments: string(args),
						},
					})
				}
			}

			choice.Message.Content = strings.Join(textParts, "")
			if len(reasoningParts) > 0 {
				choice.Message.ReasoningContent = strings.Join(reasoningParts, "")
			}
		}

		resp.Choices = append(resp.Choices, choice)
	}

	// Convert usage
	if geminiResp.UsageMetadata != nil {
		resp.Usage = &Usage{
			PromptTokens:     geminiResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: geminiResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      geminiResp.UsageMetadata.TotalTokenCount,
		}
		if geminiResp.UsageMetadata.ThoughtsTokenCount > 0 {
			resp.Usage.CompletionTokensDetails = &CompletionTokensDetails{
				ReasoningTokens: geminiResp.UsageMetadata.ThoughtsTokenCount,
			}
		}
	}

	return resp
}

// extractThinking extracts thinking content from text
func extractThinking(text string) (content string, reasoning string) {
	// Look for <vertex_think_tag> or similar thinking markers
	thinkPattern := regexp.MustCompile(`<vertex_think_tag>([\s\S]*?)</vertex_think_tag>`)
	matches := thinkPattern.FindAllStringSubmatch(text, -1)

	if len(matches) == 0 {
		return text, ""
	}

	var reasonings []string
	remaining := text

	for _, match := range matches {
		reasonings = append(reasonings, match[1])
		remaining = strings.Replace(remaining, match[0], "", 1)
	}

	return strings.TrimSpace(remaining), strings.Join(reasonings, "\n")
}

func mapFinishReason(geminiReason string) string {
	switch geminiReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY":
		return "content_filter"
	case "RECITATION":
		return "content_filter"
	case "OTHER":
		return "stop"
	default:
		if geminiReason == "" {
			return ""
		}
		return "stop"
	}
}

var toolCallCounter int64

func generateToolCallID() string {
	toolCallCounter++
	return "call_" + base64.RawURLEncoding.EncodeToString([]byte(string(rune(toolCallCounter))))[:8]
}
