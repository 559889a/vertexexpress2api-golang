package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"vertex2api-golang/internal/auth"
	"vertex2api-golang/internal/config"
	"vertex2api-golang/internal/handlers"
	"vertex2api-golang/internal/health"
	"vertex2api-golang/internal/models"
)

func main() {
	// Setup logging
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	log.Println("Starting vertex2api-golang...")

	// Load .env file (ignore error if not exists)
	if err := config.LoadEnvFile(".env"); err == nil {
		log.Println("Loaded .env file")
	}

	// Load configuration
	cfg := config.Load()

	// Validate configuration
	if len(cfg.VertexExpressAPIKeys) == 0 {
		log.Fatal("VERTEX_EXPRESS_API_KEY is required")
	}

	log.Printf("Configuration loaded: port=%s, keys=%d, roundrobin=%v, location=%s",
		cfg.AppPort, len(cfg.VertexExpressAPIKeys), cfg.RoundRobin, cfg.GCPLocation)

	// Initialize models
	models.Initialize()

	// Initialize handlers (must be after config is loaded)
	handlers.InitClient()

	// Setup routes
	mux := http.NewServeMux()

	// Health check (no auth)
	mux.HandleFunc("/health", health.Handler())

	// OpenAI compatible endpoints
	mux.HandleFunc("/v1/models", handlers.ModelsHandler)
	mux.HandleFunc("/v1/chat/completions", handlers.ChatCompletionsHandler)

	// Gemini native endpoints
	mux.HandleFunc("/gemini/v1beta/models", handlers.GeminiModelsHandler)
	mux.HandleFunc("/gemini/v1beta/", handlers.GeminiHandler)

	// Root redirect to health
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, "/health", http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})

	// Apply middleware
	handler := loggingMiddleware(corsMiddleware(auth.Middleware(mux)))

	// Create server
	server := &http.Server{
		Addr:         ":" + cfg.AppPort,
		Handler:      handler,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Server listening on port %s", cfg.AppPort)
		log.Printf("OpenAI endpoints: /v1/chat/completions, /v1/models")
		log.Printf("Gemini endpoints: /gemini/v1beta/models/{model}:generateContent")
		log.Printf("Health endpoint: /health")

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
}

// loggingMiddleware logs incoming requests
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Create response wrapper to capture status code
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rw, r)

		// Log request
		log.Printf("%s %s %d %v",
			r.Method,
			r.URL.Path,
			rw.statusCode,
			time.Since(start),
		)
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Flush implements http.Flusher for streaming support
func (rw *responseWriter) Flush() {
	if flusher, ok := rw.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// corsMiddleware handles CORS headers
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" {
			origin = "*"
		}

		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Goog-Api-Key")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		// Special handling for SSE
		if strings.Contains(r.URL.Path, "chat/completions") {
			w.Header().Set("Access-Control-Expose-Headers", "Content-Type")
		}

		next.ServeHTTP(w, r)
	})
}
