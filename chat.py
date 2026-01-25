#!/usr/bin/env python3
"""
CLI Chat Interface for AI API
Supports chat history persistence and continuation
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests

from config import API_KEY, API_URL, MODEL, MAX_TOKENS, HISTORY_DIR, ACTIVE_CHAT_FILE, INPUT_FILE, OUTPUT_FILE, STREAMING

# Ensure history directory exists
HISTORY_DIR.mkdir(exist_ok=True)


class Chat:
    """Manages chat sessions with AI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize chat manager with API key"""
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError(
                "API_KEY not found. Set it in .env file or pass it to the script."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_chat_history(self) -> Dict:
        """Create a new chat history structure"""
        now = datetime.now()
        return {
            "model": MODEL,
            "messages": [],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
                "completion_tokens": 0
            },
            "created_at": now.isoformat(),
            "last_updated_at": now.isoformat()
        }
    
    def get_chat_filename(self, created_at: str) -> str:
        """Generate filename from created_at timestamp"""
        # Remove colons from time part for filesystem safety
        timestamp = created_at.replace(":", "").replace(".", "-")
        return f"{timestamp}.json"
    
    def save_chat_history(self, chat_data: Dict, filename: Optional[str] = None) -> str:
        """Save chat history to file"""
        if filename is None:
            filename = self.get_chat_filename(chat_data["created_at"])
        
        filepath = HISTORY_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def load_chat_history(self, filename: str) -> Dict:
        """Load chat history from file"""
        filepath = HISTORY_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Chat history file not found: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_active_chat(self) -> Optional[str]:
        """Get the active chat filename"""
        if not ACTIVE_CHAT_FILE.exists():
            return None
        
        with open(ACTIVE_CHAT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("active_chat")
    
    def set_active_chat(self, filename: str):
        """Set the active chat filename"""
        with open(ACTIVE_CHAT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"active_chat": filename}, f, indent=2)
    
    def send_message(self, messages: List[Dict], stream: bool = False) -> Dict:
        """Send messages to AI API and get response"""
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": stream
        }

        try:
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=120,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def _handle_stream(self, response) -> Dict:
        """Handle streaming response from API"""
        full_content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print("\n" + "="*60)
        print("ASSISTANT:")
        print("="*60)

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                full_content += content
                        # Capture usage if present (usually in final chunk)
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        continue

        print("\n" + "="*60)

        # Return in same format as non-streaming response
        return {
            "choices": [{"message": {"content": full_content}}],
            "usage": usage
        }
    
    def update_output_md(self, chat_data: Dict):
        """Update output.md with full chat history"""
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# Chat History\n\n")
            f.write(f"**Created:** {chat_data['created_at']}\n\n")
            f.write(f"**Last Updated:** {chat_data['last_updated_at']}\n\n")
            f.write(f"**Model:** {chat_data['model']}\n\n")
            f.write("---\n\n")
            
            for msg in chat_data["messages"]:
                role = msg["role"].capitalize()
                content = msg["content"]
                f.write(f"## {role}\n\n{content}\n\n")
            
            # Add usage statistics
            usage = chat_data["usage"]
            f.write("---\n\n")
            f.write(f"**Usage Statistics:**\n\n")
            f.write(f"- Prompt Tokens: {usage['prompt_tokens']}\n")
            f.write(f"- Completion Tokens: {usage['completion_tokens']}\n")
            f.write(f"- Total Tokens: {usage['total_tokens']}\n")
    
    def chat(self, prompt: str, new_chat: bool = False, load_file: Optional[str] = None):
        """Main chat function"""
        # Determine which chat to use
        if load_file:
            # Load specified chat
            chat_data = self.load_chat_history(load_file)
            filename = load_file
            self.set_active_chat(filename)
            print(f"Loaded chat: {filename}")
            self.update_output_md(chat_data)
            return
        
        if new_chat:
            # Create new chat
            chat_data = self.create_chat_history()
            filename = None  # Will be generated on first save
        else:
            # Try to continue active chat
            active_filename = self.get_active_chat()
            if active_filename:
                try:
                    chat_data = self.load_chat_history(active_filename)
                    filename = active_filename
                    print(f"Continuing chat: {filename}")
                except FileNotFoundError:
                    print(f"Active chat file not found. Creating new chat.")
                    chat_data = self.create_chat_history()
                    filename = None
            else:
                # No active chat, create new one
                chat_data = self.create_chat_history()
                filename = None
        
        # Read prompt from input.md if no prompt provided
        if not prompt and INPUT_FILE.exists():
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        
        if not prompt:
            raise ValueError("No prompt provided. Either pass a prompt or create input.md file.")
        
        # Add user message
        chat_data["messages"].append({
            "role": "user",
            "content": prompt
        })
        
        # Send to API
        print("Sending message to AI...")
        response = self.send_message(chat_data["messages"], stream=STREAMING)

        # Extract assistant response
        assistant_message = response["choices"][0]["message"]["content"]

        # Add assistant response to chat
        chat_data["messages"].append({
            "role": "assistant",
            "content": assistant_message
        })

        # Update usage statistics (cumulative)
        response_usage = response["usage"]
        chat_data["usage"]["prompt_tokens"] += response_usage["prompt_tokens"]
        chat_data["usage"]["completion_tokens"] += response_usage["completion_tokens"]
        chat_data["usage"]["total_tokens"] += response_usage["total_tokens"]

        # Update timestamp
        chat_data["last_updated_at"] = datetime.now().isoformat()

        # Save chat history
        if filename is None:
            filename = self.save_chat_history(chat_data)
            print(f"Created new chat: {filename}")
        else:
            self.save_chat_history(chat_data, filename)
            print(f"Updated chat: {filename}")

        # Set as active chat
        self.set_active_chat(filename)

        # Update output.md
        self.update_output_md(chat_data)
        print(f"\nResponse written to {OUTPUT_FILE}")

        # Print response to console (only if not streaming, as streaming prints in real-time)
        if not STREAMING:
            print("\n" + "="*60)
            print("ASSISTANT:")
            print("="*60)
            print(assistant_message)
            print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CLI Chat Interface for AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --new "who created c++"          # Create new chat
  %(prog)s "who created python"             # Continue active chat or create new
  %(prog)s --load 2026-01-22T010757.json    # Load specific chat history
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The message to send (optional if using input.md)"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new chat (ignore active chat)"
    )
    parser.add_argument(
        "--load",
        metavar="FILENAME",
        help="Load a specific chat history file"
    )
    parser.add_argument(
        "--api-key",
        help="AI API key (overrides API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Handle --new with prompt in the argument
    if args.new and args.prompt:
        new_chat = True
        prompt = args.prompt
    elif args.new and isinstance(args.new, str):
        # Handle case where prompt follows --new
        new_chat = True
        prompt = args.new
        args.new = True
    else:
        new_chat = args.new
        prompt = args.prompt
    
    try:
        chat_manager = Chat(api_key=args.api_key)
        chat_manager.chat(
            prompt=prompt,
            new_chat=new_chat,
            load_file=args.load
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()  