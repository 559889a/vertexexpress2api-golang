package keys

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"vertex2api-golang/internal/config"
)

// AuthInfo contains authentication information for a request
type AuthInfo struct {
	ProjectID string
	APIKey    string
	Location  string
	KeyIndex  int
}

// KeyManager manages Express API keys with round-robin/random selection and retry
type KeyManager struct {
	keys         []string
	currentIndex int
	roundRobin   bool
	mu           sync.Mutex

	// Project ID cache: apiKey -> projectId
	projectCache map[string]string
	cacheMu      sync.RWMutex

	// HTTP client for discovery
	httpClient *http.Client

	// Config
	location string
}

var (
	manager *KeyManager
	once    sync.Once
)

// GetManager returns the singleton KeyManager instance
func GetManager() *KeyManager {
	once.Do(func() {
		cfg := config.Get()
		manager = &KeyManager{
			keys:         cfg.VertexExpressAPIKeys,
			currentIndex: 0,
			roundRobin:   cfg.RoundRobin,
			projectCache: make(map[string]string),
			location:     cfg.GCPLocation,
			httpClient:   createHTTPClient(cfg),
		}

		// If GCP_PROJECT_ID is set, use it for all keys
		if cfg.GCPProjectID != "" {
			for _, key := range manager.keys {
				manager.projectCache[key] = cfg.GCPProjectID
			}
		}
	})
	return manager
}

func createHTTPClient(cfg *config.Config) *http.Client {
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	}

	// Handle proxy
	if cfg.ProxyURL != "" {
		proxyURL, err := url.Parse(cfg.ProxyURL)
		if err == nil {
			transport.Proxy = http.ProxyURL(proxyURL)
		}
	}

	// Handle custom SSL cert
	if cfg.SSLCertFile != "" {
		transport.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: true, // For self-signed certs
		}
	}

	return &http.Client{
		Transport: transport,
		Timeout:   120 * time.Second,
	}
}

// PickAuth selects an API key and returns auth info
func (km *KeyManager) PickAuth(ctx context.Context) (*AuthInfo, error) {
	if len(km.keys) == 0 {
		return nil, fmt.Errorf("no Express API keys configured")
	}

	km.mu.Lock()
	var key string
	var index int

	if km.roundRobin {
		index = km.currentIndex
		key = km.keys[index]
		km.currentIndex = (km.currentIndex + 1) % len(km.keys)
	} else {
		index = rand.Intn(len(km.keys))
		key = km.keys[index]
	}
	km.mu.Unlock()

	// Get or discover project ID
	projectID, err := km.getProjectID(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("failed to get project ID: %w", err)
	}

	return &AuthInfo{
		ProjectID: projectID,
		APIKey:    key,
		Location:  km.location,
		KeyIndex:  index,
	}, nil
}

// PickAuthAtIndex picks a specific key by index
func (km *KeyManager) PickAuthAtIndex(ctx context.Context, index int) (*AuthInfo, error) {
	if len(km.keys) == 0 {
		return nil, fmt.Errorf("no Express API keys configured")
	}

	if index < 0 || index >= len(km.keys) {
		index = 0
	}

	key := km.keys[index]

	projectID, err := km.getProjectID(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("failed to get project ID: %w", err)
	}

	return &AuthInfo{
		ProjectID: projectID,
		APIKey:    key,
		Location:  km.location,
		KeyIndex:  index,
	}, nil
}

// NextKeyIndex returns the next key index for retry
func (km *KeyManager) NextKeyIndex(currentIndex int) int {
	if len(km.keys) <= 1 {
		return currentIndex
	}
	return (currentIndex + 1) % len(km.keys)
}

// KeyCount returns the number of available keys
func (km *KeyManager) KeyCount() int {
	return len(km.keys)
}

// getProjectID retrieves or discovers the project ID for a key
func (km *KeyManager) getProjectID(ctx context.Context, apiKey string) (string, error) {
	// Check cache first
	km.cacheMu.RLock()
	if projectID, ok := km.projectCache[apiKey]; ok {
		km.cacheMu.RUnlock()
		return projectID, nil
	}
	km.cacheMu.RUnlock()

	// Discover project ID
	projectID, err := km.discoverProjectID(ctx, apiKey)
	if err != nil {
		return "", err
	}

	// Cache the result
	km.cacheMu.Lock()
	km.projectCache[apiKey] = projectID
	km.cacheMu.Unlock()

	return projectID, nil
}

// discoverProjectID discovers project ID by sending an intentionally invalid request
func (km *KeyManager) discoverProjectID(ctx context.Context, apiKey string) (string, error) {
	// Send a request to a non-existent model to get the project ID from error
	url := fmt.Sprintf(
		"https://%s-aiplatform.googleapis.com/v1beta1/projects/unknown/locations/%s/publishers/google/models/gemini-1.0-pro:generateContent?key=%s",
		km.location, km.location, apiKey,
	)

	req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(`{"contents":[]}`))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := km.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Parse error response to extract project ID
	// Error message typically contains: "projects/PROJECT_ID/..."
	projectID := extractProjectIDFromError(string(body))
	if projectID == "" {
		return "", fmt.Errorf("failed to discover project ID from response: %s", string(body))
	}

	log.Printf("Discovered project ID: %s", projectID)
	return projectID, nil
}

// extractProjectIDFromError extracts project ID from Vertex error response
func extractProjectIDFromError(errorBody string) string {
	// Try to parse JSON error
	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Status  string `json:"status"`
		} `json:"error"`
	}

	if err := json.Unmarshal([]byte(errorBody), &errResp); err == nil {
		// Look for project ID in error message
		// Pattern: "projects/PROJECT_ID/" or "Project: PROJECT_ID"
		patterns := []string{
			`projects/([^/\s"]+)`,
			`Project:\s*([^\s"]+)`,
			`project[_\s]+id[:\s]+([^\s"]+)`,
		}

		for _, pattern := range patterns {
			re := regexp.MustCompile(pattern)
			if matches := re.FindStringSubmatch(errResp.Error.Message); len(matches) > 1 {
				return matches[1]
			}
		}
	}

	// Try direct pattern matching on raw body
	patterns := []string{
		`projects/([^/\s"]+)`,
		`"project":\s*"([^"]+)"`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if matches := re.FindStringSubmatch(errorBody); len(matches) > 1 {
			return matches[1]
		}
	}

	return ""
}

// GetHTTPClient returns the shared HTTP client
func (km *KeyManager) GetHTTPClient() *http.Client {
	return km.httpClient
}

// RetryConfig contains retry configuration
type RetryConfig struct {
	MaxRetries  int
	IntervalMS  int
	SwitchKey   bool // Whether to switch to next key on retry
}

// GetRetryConfig returns retry configuration from config
func GetRetryConfig() RetryConfig {
	cfg := config.Get()
	return RetryConfig{
		MaxRetries: cfg.RetryMax,
		IntervalMS: cfg.RetryIntervalMS,
		SwitchKey:  true,
	}
}
