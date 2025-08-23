"""
Ollama Client for LLM Integration

This client provides an interface to local Ollama API for natural language processing,
conversation management, and response generation in the biomedical knowledge graph Q&A system.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class OllamaModelType(Enum):
    """Available Ollama model types for different use cases."""
    LLAMA2 = "llama2"
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"
    CODELLAMA = "codellama"
    CODELLAMA_7B = "codellama:7b"
    MISTRAL = "mistral"
    NEURAL_CHAT = "neural-chat"
    STARLING = "starling-lm"

@dataclass
class ConversationMessage:
    """Structure for conversation messages."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass 
class OllamaResponse:
    """Structure for Ollama API responses."""
    content: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaClient:
    """
    Client for interacting with local Ollama API.
    
    Provides methods for model management, conversation handling,
    and response generation optimized for biomedical Q&A.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:1b",
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.conversation_history: List[ConversationMessage] = []
        self.system_prompt: Optional[str] = None
        self.session = requests.Session()
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama API."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama API")
                return True
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama API: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of available models with their information
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def set_model(self, model: str) -> bool:
        """
        Set the model to use for generation.
        
        Args:
            model: Model name to use
            
        Returns:
            True if model is available, False otherwise
        """
        available_models = self.get_available_models()
        model_names = [m.get('name', '') for m in available_models]
        
        if model in model_names:
            self.model = model
            logger.info(f"Model set to: {model}")
            return True
        else:
            logger.warning(f"Model {model} not available. Available models: {model_names}")
            return False
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set system prompt for conversation context.
        
        Args:
            system_prompt: System prompt to use for all conversations
        """
        self.system_prompt = system_prompt
        logger.info("System prompt set")
    
    def add_to_conversation(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ("system", "user", "assistant")
            content: Message content
        """
        message = ConversationMessage(role=role, content=content)
        self.conversation_history.append(message)
        
        # Limit conversation history to prevent context overflow
        max_messages = 20  # Keep last 20 messages
        if len(self.conversation_history) > max_messages:
            # Keep system messages and recent messages
            system_messages = [m for m in self.conversation_history if m.role == "system"]
            recent_messages = self.conversation_history[-max_messages:]
            self.conversation_history = system_messages + recent_messages
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_conversation_context(self) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for Ollama API.
        
        Returns:
            List of message dictionaries for API
        """
        context = []
        
        # Add system prompt if set
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for message in self.conversation_history:
            context.append({"role": message.role, "content": message.content})
        
        return context
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[ConversationMessage]] = None,
                         temperature: float = 0.7,
                         top_k: int = 40,
                         top_p: float = 0.9) -> OllamaResponse:
        """
        Generate response using Ollama API.
        
        Args:
            prompt: Input prompt
            context: Optional conversation context (uses instance history if None)
            temperature: Sampling temperature (0.0 to 1.0)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            OllamaResponse object with generated content
        """
        # Prepare messages
        messages = []
        
        if context is not None:
            messages.extend([{"role": m.role, "content": m.content} for m in context])
        else:
            messages = self.get_conversation_context()
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API request
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            "stream": False
        }
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating response with model {self.model} (attempt {attempt + 1})")
                
                # Try generate API (compatible with this Ollama version)
                generate_payload = {
                    "model": self.model,
                    "prompt": messages[-1]["content"] if messages else prompt,
                    "stream": False,
                    "options": payload.get("options", {})
                }
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=generate_payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Convert generate response to chat format
                gen_data = response.json()
                data = {
                    "model": gen_data.get("model", self.model),
                    "message": {"content": gen_data.get("response", "")},
                    "done": gen_data.get("done", True),
                    "total_duration": gen_data.get("total_duration"),
                    "load_duration": gen_data.get("load_duration"),
                    "prompt_eval_count": gen_data.get("prompt_eval_count"),
                    "eval_count": gen_data.get("eval_count"),
                    "eval_duration": gen_data.get("eval_duration")
                }
                
                # Parse response
                assistant_message = data.get('message', {})
                content = assistant_message.get('content', '')
                
                ollama_response = OllamaResponse(
                    content=content,
                    model=data.get('model', self.model),
                    done=data.get('done', True),
                    total_duration=data.get('total_duration'),
                    load_duration=data.get('load_duration'),
                    prompt_eval_count=data.get('prompt_eval_count'),
                    eval_count=data.get('eval_count'),
                    eval_duration=data.get('eval_duration')
                )
                
                # Add to conversation history
                self.add_to_conversation("user", prompt)
                self.add_to_conversation("assistant", content)
                
                logger.info(f"Response generated successfully ({len(content)} chars)")
                return ollama_response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception(f"Failed to generate response after {self.max_retries} attempts")
    
    def stream_response(self, 
                       prompt: str,
                       context: Optional[List[ConversationMessage]] = None,
                       temperature: float = 0.7,
                       top_k: int = 40,
                       top_p: float = 0.9) -> Iterator[str]:
        """
        Generate streaming response using Ollama API.
        
        Args:
            prompt: Input prompt
            context: Optional conversation context
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Yields:
            Chunks of generated text
        """
        # Prepare messages
        messages = []
        
        if context is not None:
            messages.extend([{"role": m.role, "content": m.content} for m in context])
        else:
            messages = self.get_conversation_context()
        
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API request
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            },
            "stream": True
        }
        
        try:
            logger.info(f"Starting streaming response with model {self.model}")
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            full_content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data:
                            chunk = data['message'].get('content', '')
                            if chunk:
                                full_content += chunk
                                yield chunk
                        
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Add to conversation history
            self.add_to_conversation("user", prompt)
            self.add_to_conversation("assistant", full_content)
            
            logger.info(f"Streaming response completed ({len(full_content)} chars)")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Streaming request failed: {e}")
            raise
    
    def generate_with_knowledge_context(self, 
                                      question: str,
                                      knowledge_data: Dict[str, Any],
                                      instruction_template: str = None) -> OllamaResponse:
        """
        Generate response incorporating knowledge graph data.
        
        Args:
            question: User question
            knowledge_data: Retrieved knowledge from graph
            instruction_template: Optional template for formatting the prompt
            
        Returns:
            OllamaResponse with knowledge-grounded answer
        """
        # Default template for biomedical knowledge integration
        if instruction_template is None:
            instruction_template = """Based on the following biomedical knowledge graph data, please answer the user's question. 
            Be sure to cite specific information from the provided data and explain your reasoning.

Knowledge Graph Data:
{knowledge_data}

User Question: {question}

Please provide a comprehensive answer based on the provided scientific data."""

        # Format knowledge data for readability
        formatted_knowledge = self._format_knowledge_for_llm(knowledge_data)
        
        # Create prompt with knowledge context
        prompt = instruction_template.format(
            knowledge_data=formatted_knowledge,
            question=question
        )
        
        return self.generate_response(prompt)
    
    def _format_knowledge_for_llm(self, knowledge_data: Dict[str, Any]) -> str:
        """
        Format knowledge graph data for LLM consumption.
        
        Args:
            knowledge_data: Raw knowledge data from graph queries
            
        Returns:
            Formatted string representation
        """
        formatted_parts = []
        
        for key, value in knowledge_data.items():
            if isinstance(value, list) and value:
                formatted_parts.append(f"{key.title()}:")
                for i, item in enumerate(value[:10]):  # Limit to first 10 items
                    if isinstance(item, dict):
                        formatted_parts.append(f"  {i+1}. {self._format_dict_item(item)}")
                    else:
                        formatted_parts.append(f"  {i+1}. {item}")
                if len(value) > 10:
                    formatted_parts.append(f"  ... and {len(value) - 10} more items")
                formatted_parts.append("")
                
            elif isinstance(value, dict) and value:
                formatted_parts.append(f"{key.title()}:")
                formatted_parts.append(f"  {self._format_dict_item(value)}")
                formatted_parts.append("")
                
            elif value not in [None, [], {}, ""]:
                formatted_parts.append(f"{key.title()}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _format_dict_item(self, item: Dict[str, Any]) -> str:
        """Format a dictionary item for display."""
        key_order = ['name', 'gene', 'go_term', 'disease', 'drug', 'type', 'namespace', 'definition']
        
        parts = []
        # Add important keys first
        for key in key_order:
            if key in item and item[key] not in [None, "", []]:
                parts.append(f"{key}: {item[key]}")
        
        # Add remaining keys
        for key, value in item.items():
            if key not in key_order and value not in [None, "", [], {}]:
                if isinstance(value, (dict, list)):
                    parts.append(f"{key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else '?'} items")
                else:
                    parts.append(f"{key}: {value}")
        
        return " | ".join(parts[:5])  # Limit to first 5 attributes
    
    def generate_follow_up_questions(self, 
                                   question: str, 
                                   answer: str,
                                   knowledge_data: Dict[str, Any]) -> List[str]:
        """
        Generate relevant follow-up questions based on the conversation.
        
        Args:
            question: Original question
            answer: Generated answer
            knowledge_data: Knowledge graph data used
            
        Returns:
            List of suggested follow-up questions
        """
        prompt = f"""Based on this biomedical Q&A exchange, suggest 3-5 specific follow-up questions that would help the user explore related aspects of this topic in more depth.

Original Question: {question}

Answer Provided: {answer[:500]}...

Available Related Data: {list(knowledge_data.keys())}

Please generate follow-up questions that:
1. Explore mechanisms or pathways mentioned
2. Ask about related genes, diseases, or drugs  
3. Investigate clinical or research applications
4. Focus on specific aspects that could be analyzed further

Format as a simple numbered list."""

        try:
            response = self.generate_response(prompt)
            # Parse numbered list from response
            lines = response.content.split('\n')
            questions = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                    # Clean up formatting
                    question = line.split('.', 1)[-1].strip()
                    if question and question.endswith('?'):
                        questions.append(question)
            
            return questions[:5]  # Limit to 5 questions
            
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions: {e}")
            return []
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics and status."""
        return {
            "base_url": self.base_url,
            "current_model": self.model,
            "conversation_length": len(self.conversation_history),
            "system_prompt_set": self.system_prompt is not None,
            "available_models": len(self.get_available_models()),
            "connection_status": self._test_connection()
        }