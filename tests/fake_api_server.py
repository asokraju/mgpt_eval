#!/usr/bin/env python3
"""
Fake API server for testing the medical claims evaluation pipeline.

This server simulates the MediClaimGPT API endpoints with realistic responses
for testing purposes. It provides embeddings and text generation capabilities
with configurable delays and error injection for robust testing.

Usage:
    python fake_api_server.py --port 8001
    python fake_api_server.py --port 8001 --delay 0.1 --error-rate 0.05
"""

import argparse
import json
import random
import time
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime
import threading
import logging

# Suppress Flask logging in quiet mode
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

app = Flask(__name__)

# Global configuration
CONFIG = {
    "embedding_dim": 128,
    "max_tokens": 2048,
    "delay": 0.0,
    "error_rate": 0.0,
    "vocab_size": 10000
}

# Medical code vocabulary for realistic generation
MEDICAL_CODES = [
    "E119", "Z03818", "N6320", "K9289", "76642", "O0903", "U0003",
    "G0378", "Z91048", "M1710", "N6322", "Z09", "Z1239", "O9989",
    "E1169", "Z0000", "N3940", "K5990", "76700", "O8020", "U0004",
    "G0379", "Z91049", "M1711", "N6323", "Z10", "Z1240", "O9990"
]

def generate_random_embedding(dim=128):
    """Generate a random embedding vector."""
    # Use normal distribution centered around 0 with std=0.5
    return np.random.normal(0, 0.5, dim).tolist()

def generate_medical_sequence(prompt="", max_tokens=50):
    """Generate a realistic medical code sequence."""
    # Start with prompt if provided
    if prompt and not prompt.endswith(" "):
        prompt += " "
    
    # Generate sequence
    codes = []
    remaining_tokens = max_tokens
    
    while remaining_tokens > 0 and len(codes) < 20:  # Limit sequence length
        if random.random() < 0.7:  # 70% chance of medical code
            code = random.choice(MEDICAL_CODES)
            codes.append(code)
            remaining_tokens -= 1
        else:  # 30% chance of separator
            if codes and codes[-1] != "|eoc|":
                codes.append("|eoc|")
                remaining_tokens -= 1
    
    # Ensure sequence ends properly
    if codes and codes[-1] != "|eoc|":
        codes.append("|eoc|")
    
    generated = " ".join(codes)
    return prompt + generated

def simulate_delay():
    """Simulate API processing delay."""
    if CONFIG["delay"] > 0:
        time.sleep(CONFIG["delay"])

def maybe_inject_error():
    """Randomly inject errors based on error rate."""
    if random.random() < CONFIG["error_rate"]:
        error_types = [
            ("timeout", 408, "Request timeout"),
            ("server_error", 500, "Internal server error"),
            ("bad_gateway", 502, "Bad gateway"),
            ("service_unavailable", 503, "Service temporarily unavailable")
        ]
        error_type, status_code, message = random.choice(error_types)
        return status_code, {"error": error_type, "message": message}
    return None

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_dim": CONFIG["embedding_dim"],
            "max_tokens": CONFIG["max_tokens"],
            "delay": CONFIG["delay"],
            "error_rate": CONFIG["error_rate"]
        }
    })

@app.route("/embeddings", methods=["POST"])
def single_embedding():
    """Single embedding endpoint."""
    simulate_delay()
    
    # Check for error injection
    error = maybe_inject_error()
    if error:
        status_code, error_response = error
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        claims = data.get("claims", [])
        
        if not claims:
            return jsonify({"error": "No claims provided"}), 400
        
        # Take only first claim for single endpoint
        claim = claims[0] if isinstance(claims, list) else claims
        
        # Generate embedding
        embedding = generate_random_embedding(CONFIG["embedding_dim"])
        input_tokens = len(claim.split()) if isinstance(claim, str) else 10
        
        response = {
            "embeddings": [embedding],
            "input_tokens": [input_tokens],
            "embedding_dim": CONFIG["embedding_dim"],
            "execution_time": CONFIG["delay"] + random.uniform(0.01, 0.05)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/embeddings_batch", methods=["POST"])
def batch_embeddings():
    """Batch embeddings endpoint."""
    simulate_delay()
    
    # Check for error injection
    error = maybe_inject_error()
    if error:
        status_code, error_response = error
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        claims = data.get("claims", [])
        batch_size = data.get("batch_size", len(claims))
        
        if not claims:
            return jsonify({"error": "No claims provided"}), 400
        
        # Process up to batch_size claims
        processed_claims = claims[:batch_size]
        
        # Generate embeddings
        embeddings = []
        input_tokens = []
        
        for claim in processed_claims:
            embedding = generate_random_embedding(CONFIG["embedding_dim"])
            tokens = len(claim.split()) if isinstance(claim, str) else 10
            
            embeddings.append(embedding)
            input_tokens.append(tokens)
        
        response = {
            "embeddings": embeddings,
            "input_tokens": input_tokens,
            "embedding_dim": CONFIG["embedding_dim"],
            "execution_time": CONFIG["delay"] + random.uniform(0.05, 0.15),
            "batch_size": len(embeddings)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate_text():
    """Text generation endpoint."""
    simulate_delay()
    
    # Check for error injection
    error = maybe_inject_error()
    if error:
        status_code, error_response = error
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 50)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 50)
        seed = data.get("seed", None)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate text
        generated_text = generate_medical_sequence(prompt, max_new_tokens)
        tokens_generated = len(generated_text.replace(prompt, "").split())
        
        response = {
            "generated_text": generated_text,
            "tokens_generated": tokens_generated,
            "prompt_tokens": len(prompt.split()),
            "total_tokens": len(generated_text.split()),
            "execution_time": CONFIG["delay"] + random.uniform(0.1, 0.3),
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "seed": seed
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_batch", methods=["POST"])
def generate_batch():
    """Batch text generation endpoint."""
    simulate_delay()
    
    # Check for error injection
    error = maybe_inject_error()
    if error:
        status_code, error_response = error
        return jsonify(error_response), status_code
    
    try:
        data = request.get_json()
        prompts = data.get("prompts", [])
        max_new_tokens = data.get("max_new_tokens", 50)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 50)
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400
        
        # Generate text for each prompt
        results = []
        for i, prompt in enumerate(prompts):
            # Use different seed for each prompt for variety
            if data.get("seed") is not None:
                random.seed(data["seed"] + i)
                np.random.seed(data["seed"] + i)
            
            generated_text = generate_medical_sequence(prompt, max_new_tokens)
            tokens_generated = len(generated_text.replace(prompt, "").split())
            
            results.append({
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "prompt_tokens": len(prompt.split()),
                "total_tokens": len(generated_text.split())
            })
        
        response = {
            "results": results,
            "batch_size": len(results),
            "execution_time": CONFIG["delay"] + random.uniform(0.2, 0.5),
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/config", methods=["GET", "POST"])
def manage_config():
    """Get or update server configuration."""
    if request.method == "GET":
        return jsonify(CONFIG)
    
    elif request.method == "POST":
        try:
            updates = request.get_json()
            
            # Validate and update configuration
            for key, value in updates.items():
                if key in CONFIG:
                    if key == "embedding_dim" and (value < 1 or value > 2048):
                        return jsonify({"error": f"Invalid embedding_dim: {value}"}), 400
                    elif key == "delay" and (value < 0 or value > 10):
                        return jsonify({"error": f"Invalid delay: {value}"}), 400
                    elif key == "error_rate" and (value < 0 or value > 1):
                        return jsonify({"error": f"Invalid error_rate: {value}"}), 400
                    
                    CONFIG[key] = value
                else:
                    return jsonify({"error": f"Unknown config key: {key}"}), 400
            
            return jsonify({"message": "Configuration updated", "config": CONFIG})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get server statistics."""
    return jsonify({
        "requests_served": getattr(app, '_request_count', 0),
        "uptime_seconds": time.time() - getattr(app, '_start_time', time.time()),
        "medical_codes_available": len(MEDICAL_CODES),
        "config": CONFIG
    })

@app.before_request
def before_request():
    """Count requests for statistics."""
    if not hasattr(app, '_request_count'):
        app._request_count = 0
        app._start_time = time.time()
    app._request_count += 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fake API server for medical claims evaluation testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start basic server
  python fake_api_server.py
  
  # Start with custom port and delay
  python fake_api_server.py --port 8001 --delay 0.1
  
  # Start with error injection for testing
  python fake_api_server.py --error-rate 0.05 --delay 0.2
  
  # Start in quiet mode
  python fake_api_server.py --quiet
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Artificial delay for each request in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.0,
        help="Probability of injecting errors (0.0-1.0, default: 0.0)"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - minimal logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode - verbose logging"
    )
    
    args = parser.parse_args()
    
    # Update global configuration
    CONFIG["delay"] = args.delay
    CONFIG["error_rate"] = args.error_rate
    CONFIG["embedding_dim"] = args.embedding_dim
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        app.debug = True
    
    # Print startup info
    if not args.quiet:
        print(f"🚀 Starting Fake MediClaimGPT API Server")
        print(f"   URL: http://{args.host}:{args.port}")
        print(f"   Embedding Dim: {CONFIG['embedding_dim']}")
        print(f"   Delay: {CONFIG['delay']}s")
        print(f"   Error Rate: {CONFIG['error_rate'] * 100:.1f}%")
        print(f"   Available endpoints:")
        print(f"     GET  /health - Health check")
        print(f"     POST /embeddings - Single embedding")
        print(f"     POST /embeddings_batch - Batch embeddings")
        print(f"     POST /generate - Text generation")
        print(f"     POST /generate_batch - Batch generation")
        print(f"     GET  /stats - Server statistics")
        print(f"     GET/POST /config - Configuration management")
        print()
        print(f"   To test: curl http://{args.host}:{args.port}/health")
        print(f"   To stop: Ctrl+C")
        print()
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())