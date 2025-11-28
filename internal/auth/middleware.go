package auth

import (
	"encoding/json"
	"net/http"
	"strings"

	"vertex2api-golang/internal/config"
)

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Middleware validates API key authentication
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cfg := config.Get()

		// Skip auth if no API key configured
		if cfg.APIKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		// Extract API key from various sources
		apiKey := extractAPIKey(r)

		if apiKey == "" || apiKey != cfg.APIKey {
			sendAuthError(w, "Invalid API key")
			return
		}

		next.ServeHTTP(w, r)
	})
}

// extractAPIKey extracts API key from request
// Supports: Authorization Bearer, x-goog-api-key header, URL query param
func extractAPIKey(r *http.Request) string {
	// Check Authorization header (Bearer token)
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}

	// Check x-goog-api-key header (Gemini style)
	if key := r.Header.Get("x-goog-api-key"); key != "" {
		return key
	}

	// Check URL query parameter
	if key := r.URL.Query().Get("key"); key != "" {
		return key
	}

	return ""
}

func sendAuthError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)

	resp := ErrorResponse{}
	resp.Error.Message = message
	resp.Error.Type = "invalid_request_error"
	resp.Error.Code = "invalid_api_key"

	json.NewEncoder(w).Encode(resp)
}
