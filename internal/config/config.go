package config

import (
	"os"
	"strconv"
	"strings"
)

// Config holds all application configuration
type Config struct {
	// Server
	AppPort string

	// Authentication
	APIKey string

	// Vertex Express Keys
	VertexExpressAPIKeys []string
	RoundRobin           bool

	// GCP Settings
	GCPProjectID string
	GCPLocation  string

	// Retry Settings
	RetryMax        int
	RetryIntervalMS int

	// Models
	ModelsConfigURL string

	// Proxy & TLS
	ProxyURL    string
	SSLCertFile string

	// Features
	SafetyScore bool
}

var cfg *Config

// Load parses environment variables and returns Config
func Load() *Config {
	if cfg != nil {
		return cfg
	}

	cfg = &Config{
		AppPort:              getEnv("APP_PORT", "8080"),
		APIKey:               getEnv("API_KEY", ""),
		VertexExpressAPIKeys: parseKeys(getEnv("VERTEX_EXPRESS_API_KEY", "")),
		RoundRobin:           getEnvBool("ROUNDROBIN", false),
		GCPProjectID:         getEnv("GCP_PROJECT_ID", ""),
		GCPLocation:          getEnv("GCP_LOCATION", "global"),
		RetryMax:             getEnvInt("RETRY_MAX", 3),
		RetryIntervalMS:      getEnvInt("RETRY_INTERVAL_MS", 1000),
		ModelsConfigURL:      getEnv("MODELS_CONFIG_URL", ""),
		ProxyURL:             getEnv("PROXY_URL", ""),
		SSLCertFile:          getEnv("SSL_CERT_FILE", ""),
		SafetyScore:          getEnvBool("SAFETY_SCORE", false),
	}

	return cfg
}

// Get returns the current config (must call Load first)
func Get() *Config {
	if cfg == nil {
		return Load()
	}
	return cfg
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvBool(key string, defaultVal bool) bool {
	val := os.Getenv(key)
	if val == "" {
		return defaultVal
	}
	val = strings.ToLower(val)
	return val == "true" || val == "1" || val == "yes"
}

func getEnvInt(key string, defaultVal int) int {
	val := os.Getenv(key)
	if val == "" {
		return defaultVal
	}
	if i, err := strconv.Atoi(val); err == nil {
		return i
	}
	return defaultVal
}

func parseKeys(s string) []string {
	if s == "" {
		return nil
	}
	keys := strings.Split(s, ",")
	result := make([]string, 0, len(keys))
	for _, k := range keys {
		k = strings.TrimSpace(k)
		if k != "" {
			result = append(result, k)
		}
	}
	return result
}
