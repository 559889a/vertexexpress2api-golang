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

	"vertex2api-golang/internal/models"
)

// modelActionPattern parses Gemini API path format: models/{model}:{action}
var modelActionPattern = regexp.MustCompile(`^models/([^:]+):(.+)$`)

// geminiModel represents a model in the Gemini API format
type geminiModel struct {
	Name        string `json:"name"`
	DisplayName string `json:"displayName"`
}

// geminiModelsResponse represents the models list response
type geminiModelsResponse struct {
	Models []geminiModel `json:"models"`
}

// GeminiHandler handles /gemini/v1beta/* endpoints
func GeminiHandler(w http.ResponseWriter, r *http.Request) {
	// Extract model and action from path
	// Path format: /gemini/v1beta/models/{model}:{action}
	path := strings.TrimPrefix(r.URL.Path, "/gemini/v1beta/")

	// Parse model and action using pre-compiled pattern
	matches := modelActionPattern.FindStringSubmatch(path)

	if len(matches) != 3 {
		sendError(w, http.StatusBadRequest, "invalid_request", "Invalid path format. Expected: /gemini/v1beta/models/{model}:{action}")
		return
	}

	model := matches[1]
	action := matches[2]

	log.Printf("GeminiHandler: model=%s, action=%s", model, action)

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, http.StatusBadRequest, "invalid_request", "Failed to read request body")
		return
	}
	defer r.Body.Close()

	log.Printf("GeminiHandler request body: %s", string(body))

	// Get auth info
	ctx := r.Context()
	auth, err := keyManager.PickAuth(ctx)
	if err != nil {
		sendError(w, http.StatusInternalServerError, "server_error", "Failed to get auth: "+err.Error())
		return
	}

	// Determine location - gemini-2.5/3 models require "global"
	location := auth.Location
	if strings.Contains(model, "gemini-2.5") || strings.Contains(model, "gemini-3") {
		location = "global"
	}

	// Build Gemini native endpoint URL
	// Format: https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:{action}?key={key}
	url := fmt.Sprintf(
		"https://aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:%s?key=%s",
		auth.ProjectID,
		location,
		model,
		action,
		auth.APIKey,
	)

	// For streaming, add alt=sse parameter
	if action == "streamGenerateContent" {
		url += "&alt=sse"
	}

	log.Printf("GeminiHandler URL: %s", strings.Replace(url, auth.APIKey, "***", 1))

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		sendError(w, http.StatusInternalServerError, "server_error", "Failed to create request")
		return
	}
	req.Header.Set("Content-Type", "application/json")

	// For streaming, set Accept header
	if action == "streamGenerateContent" {
		req.Header.Set("Accept", "text/event-stream")
	}

	// Forward request
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("GeminiHandler error: %v", err)
		sendError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	defer resp.Body.Close()

	log.Printf("GeminiHandler response status: %d", resp.StatusCode)

	// If error status, forward the error response to client
	if resp.StatusCode != http.StatusOK {
		// Read error response; ignore read errors as we're already on error path
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("GeminiHandler error response: %s", string(respBody))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	// Handle streaming response
	if action == "streamGenerateContent" {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(resp.StatusCode)

		flusher, ok := w.(http.Flusher)
		if !ok {
			log.Printf("GeminiHandler: Flusher not available, falling back to io.Copy")
			io.Copy(w, resp.Body)
			return
		}

		// Stream response
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

		lineCount := 0
		for scanner.Scan() {
			line := scanner.Text()
			lineCount++
			w.Write([]byte(line + "\n"))
			flusher.Flush()
		}

		if err := scanner.Err(); err != nil {
			log.Printf("GeminiHandler stream scanner error: %v", err)
		}

		log.Printf("GeminiHandler stream completed, lines: %d", lineCount)
	} else {
		// Non-streaming response - copy headers then body
		for key, values := range resp.Header {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
		w.WriteHeader(resp.StatusCode)
		n, _ := io.Copy(w, resp.Body)
		log.Printf("GeminiHandler non-streaming response, bytes: %d", n)
	}
}

// GeminiModelsHandler handles /gemini/v1beta/models endpoint
func GeminiModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Method not allowed")
		return
	}

	// Get models from shared models package
	modelsList := models.GetModels()

	resp := geminiModelsResponse{
		Models: make([]geminiModel, 0, len(modelsList)),
	}

	for _, m := range modelsList {
		resp.Models = append(resp.Models, geminiModel{
			Name:        "models/" + m.ID,
			DisplayName: m.ID,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}
