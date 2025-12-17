package models

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"vertex2api-golang/internal/config"
)

// Model represents an OpenAI-style model
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	Root    string `json:"root,omitempty"`
	Parent  string `json:"parent,omitempty"`
}

// ModelsResponse is the OpenAI-style models list response
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ModelAlias defines model alias with special configurations
type ModelAlias struct {
	Target        string `json:"target"`
	ThinkingLevel string `json:"thinking_level,omitempty"` // "high" or "low"
}

var (
	modelList    []Model
	modelAliases map[string]ModelAlias
	modelMu      sync.RWMutex
	initialized  bool
)

// VertexModelsConfig represents the JSON config file structure
type VertexModelsConfig struct {
	VertexModels        []string `json:"vertex_models"`
	VertexExpressModels []string `json:"vertex_express_models"`
}

// Default models list (Vertex Express compatible)
var defaultModels = []string{
	"gemini-2.0-flash",
	"gemini-2.0-flash-001",
	"gemini-2.0-flash-lite",
	"gemini-2.0-flash-lite-001",
	"gemini-2.5-flash",
	"gemini-2.5-flash-image",
	"gemini-2.5-flash-image-preview",
	"gemini-2.5-flash-lite-preview-09-2025",
	"gemini-2.5-flash-preview-09-2025",
	"gemini-2.5-pro",
	"gemini-3-flash-preview",
	"gemini-3-pro-image-preview",
	"gemini-3-pro-preview",
}

// Default aliases with thinking levels
var defaultAliases = map[string]ModelAlias{
	"gemini-3-pro-preview-high": {
		Target:        "gemini-3-pro-preview",
		ThinkingLevel: "high",
	},
	"gemini-3-pro-preview-low": {
		Target:        "gemini-3-pro-preview",
		ThinkingLevel: "low",
	},
}

// Initialize loads models from config or uses defaults
func Initialize() {
	modelMu.Lock()
	defer modelMu.Unlock()

	if initialized {
		return
	}

	cfg := config.Get()
	models := loadModels(cfg.ModelsConfigURL)

	modelList = make([]Model, 0, len(models)+len(defaultAliases))
	now := time.Now().Unix()

	// Add base models
	for _, m := range models {
		modelList = append(modelList, Model{
			ID:      m,
			Object:  "model",
			Created: now,
			OwnedBy: "google",
			Root:    m,
		})
	}

	// Add aliases
	modelAliases = make(map[string]ModelAlias)
	for alias, target := range defaultAliases {
		modelAliases[alias] = target
		modelList = append(modelList, Model{
			ID:      alias,
			Object:  "model",
			Created: now,
			OwnedBy: "google",
			Root:    target.Target,
		})
	}

	initialized = true
	log.Printf("Loaded %d models (including %d aliases)", len(modelList), len(modelAliases))
}

func loadModels(configURL string) []string {
	// Try loading from local file first
	if data, err := os.ReadFile("vertexModels.json"); err == nil {
		if models := parseModelsJSON(data); models != nil {
			log.Println("Loaded models from vertexModels.json")
			return models
		}
	}

	// Try loading from URL if configured
	if configURL != "" {
		resp, err := http.Get(configURL)
		if err == nil {
			defer resp.Body.Close()
			data, err := io.ReadAll(resp.Body)
			if err == nil {
				if models := parseModelsJSON(data); models != nil {
					log.Printf("Loaded models from %s", configURL)
					return models
				}
			}
		}
	}

	// Use defaults
	log.Println("Using default models list")
	return defaultModels
}

// parseModelsJSON parses models from JSON, supporting both formats:
// 1. Simple array: ["model1", "model2"]
// 2. Object with vertex_express_models: {"vertex_express_models": ["model1", "model2"]}
func parseModelsJSON(data []byte) []string {
	// Try object format first (with vertex_express_models)
	var config VertexModelsConfig
	if err := json.Unmarshal(data, &config); err == nil {
		if len(config.VertexExpressModels) > 0 {
			return config.VertexExpressModels
		}
		if len(config.VertexModels) > 0 {
			return config.VertexModels
		}
	}

	// Try simple array format
	var models []string
	if err := json.Unmarshal(data, &models); err == nil && len(models) > 0 {
		return models
	}

	return nil
}

// GetModels returns all available models
func GetModels() []Model {
	modelMu.RLock()
	defer modelMu.RUnlock()

	if !initialized {
		modelMu.RUnlock()
		Initialize()
		modelMu.RLock()
	}

	return modelList
}

// GetModelsResponse returns OpenAI-style models response
func GetModelsResponse() ModelsResponse {
	return ModelsResponse{
		Object: "list",
		Data:   GetModels(),
	}
}

// ResolveModel resolves alias to actual model and returns config
func ResolveModel(modelID string) (string, *ModelAlias) {
	modelMu.RLock()
	defer modelMu.RUnlock()

	if alias, ok := modelAliases[modelID]; ok {
		return alias.Target, &alias
	}
	return modelID, nil
}
