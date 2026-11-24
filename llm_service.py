import os
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import logging
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


from utils import setup_logger, validate_api_key, calculate_token_count


@dataclass
class LLMConfig:
    api_key: str
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.1  
    max_tokens: int = 1024
    provider: str = "groq"  
    streaming: bool = True


class LLMService:
    def __init__(self, config: LLMConfig):
        self.logger = setup_logger("llm_service")
        self.config = config
        self.client = None
        
        if not validate_api_key(config.api_key):
            raise ValueError("Invalid API key provided")
        
        if config.provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq library not installed. Install with: pip install groq")
            self.client = Groq(api_key=config.api_key)
            self.logger.info(f"Initialized Groq client")
            
            
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    def _build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        context_text = "\n\n---\n\n".join([ 
            f"Source: {chunk['metadata'].get('source', 'Unknown')} (Page {chunk['metadata'].get('page', 'N/A')})\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-3:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_context += f"{role.capitalize()}: {content}\n"
        
        prompt = f"""You are an intelligent document assistant. Answer questions based ONLY on the provided context from documents.

Context from documents:
{context_text}

{history_context}

User Question: {query}

Instructions:
1. Answer the question using ONLY the information provided in the context above.
2. If the answer is not in the context, say "I cannot find this information in the provided documents."
3. Be specific and cite the sources used in your answer.
4. Do not make up information or use outside knowledge.
5. Provide a clear, concise answer.
6. If multiple sources provide relevant information, synthesize them.

Answer:"""
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        self.logger.info(f"Generating answer for query: {query[:50]}...")
        
        prompt = self._build_prompt(query, context_chunks, conversation_history)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful document assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                answer = response.choices[0].message.content
                
                prompt_tokens = calculate_token_count(prompt)
                completion_tokens = calculate_token_count(answer)
                
                result = {
                    "answer": answer,
                    "sources": [chunk["metadata"] for chunk in context_chunks],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "model": self.config.model
                }
                
                self.logger.info(f"Generated answer with {completion_tokens} tokens")
                return result
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg[:100]}...")
                
                if "resource_exhausted" in error_msg or "unavailable" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)
                        self.logger.info(f"API overloaded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error("API service is currently unavailable. Please try again later.")
                        raise Exception("The AI service is temporarily overloaded. Please try again in a few minutes.")
                else:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        self.logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Error generating answer after {max_retries} attempts: {e}")
                        raise
    
    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        self.logger.info(f"Generating streaming answer for query: {query[:50]}...")
        
        prompt = self._build_prompt(query, context_chunks, conversation_history)
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful document assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in streaming generation: {error_msg[:100]}...")
            
            if "resource_exhausted" in error_msg or "unavailable" in error_msg:
                raise Exception("The AI service is temporarily overloaded. Please try again in a few minutes.")
            raise
    
    def re_rank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        self.logger.info(f"Re-ranking {len(results)} results")
        
        results_text = "\n\n".join([
            f"Result {i+1}: {result['text'][:200]}..."
            for i, result in enumerate(results)
        ])
        
        prompt = f"""Given the query: "{query}"

Rank the following results by relevance (1 = most relevant, {len(results)} = least relevant):

{results_text}

Return only the ranking numbers in order, separated by commas."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            ranking_str = response.choices[0].message.content.strip()
            rankings = [int(x.strip()) for x in ranking_str.split(',')]
            
            ranked_results = [results[i-1] for i in rankings]
            
            self.logger.info(f"Re-ranking complete, returning top {top_k}")
            return ranked_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in re-ranking: {e}")
            return results[:top_k]
