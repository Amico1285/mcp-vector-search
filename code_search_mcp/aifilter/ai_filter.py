"""AI-powered search result filtering using Claude CLI."""
import os
import json
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Setup logger for AI filter
logger = logging.getLogger('code_searcher.ai_filter')
# Use parent logger's configuration


class AIFilter:
    """Filter search results using Claude CLI for relevance evaluation."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",  # Sonnet 4 model for better understanding
        timeout_seconds: int = 120,
        prompt_file: str = "v4_precision.md"
    ):
        """
        Initialize AI filter with Claude CLI.
        
        Args:
            model: Claude model to use for filtering
            timeout_seconds: Timeout for Claude CLI calls
            prompt_file: Name of the prompt file to use (in prompts/ directory)
        """
        self.model = model
        self.timeout_seconds = timeout_seconds
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_file)
        
        # Check if claude CLI is available
        self._check_claude_cli()
        logger.info(f"[AI_FILTER] Initialized with model: {model}, timeout: {timeout_seconds}s, prompt: {prompt_file}")
    
    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from MD file."""
        prompt_path = Path(__file__).parent / "prompts" / prompt_file
        
        # Fallback to default prompt if file not found
        if not prompt_path.exists():
            logger.warning(f"[AI_FILTER] Prompt file not found: {prompt_path}, using default prompt")
            return self._get_default_prompt()
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"[AI_FILTER] Loaded prompt template from {prompt_path}")
                return content
        except Exception as e:
            logger.error(f"[AI_FILTER] Error loading prompt template: {e}, using default prompt")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template as fallback."""
        return """You are helping improve search results for a codebase search tool. Analyze the search results and identify which files are truly relevant to the user's query based on file content and context.

The user searched for: "{query}"

Below are {num_results} search results. Analyze them and identify which files are truly relevant to the user's query.

{results_text}

Return a JSON with the numbers of relevant files:
{{
  "relevant_indices": [1, 2, 5, 8]  // use the numbers shown in square brackets above
}}

If no files are relevant, return:
{{
  "relevant_indices": []
}}

Be selective - only include files that truly match what the user is searching for. Consider:
- Does the file content match the query intent?
- Is this the type of file the user is looking for?
- Does it contain relevant code/documentation?

It's perfectly fine to return an empty list if nothing matches well."""
    
    def _check_claude_cli(self):
        """Check if claude CLI is available."""
        try:
            result = subprocess.run(
                ['claude', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("Claude CLI not found. Please install it first.")
        except FileNotFoundError:
            raise RuntimeError("Claude CLI not found. Please install it first.")
    
    def filter_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        return_all_with_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Filter search results using Claude CLI for relevance evaluation.
        
        Args:
            query: Original search query
            results: List of search results to filter
            return_all_with_scores: If True, return all results with relevance scores (not implemented)
            
        Returns:
            Filtered list of relevant results
        """
        if not results:
            return []
        
        # Prepare the prompt for Claude
        prompt = self._build_evaluation_prompt(query, results)
        logger.info(f"[AI_FILTER] Processing {len(results)} results for query: '{query}'")
        
        # Log input to aifilter.log for debugging
        self._log_to_file("INPUT", prompt, query)
        
        try:
            # Create temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(prompt)
                temp_file_path = temp_file.name
            
            # Use JSON output format for structured response
            cmd = f'cat "{temp_file_path}" | claude --model {self.model} --output-format json'
            
            try:
                result_proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
            except subprocess.TimeoutExpired:
                # Clean up temp file
                os.unlink(temp_file_path)
                logger.warning(f"[AI_FILTER] Claude CLI timed out after {self.timeout_seconds}s, returning original results")
                self._log_to_file("ERROR", "Timeout", query)
                return results
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Log raw output for debugging
            self._log_to_file("OUTPUT", f"Return code: {result_proc.returncode}\nStdout: {result_proc.stdout}\nStderr: {result_proc.stderr}", query)
            
            if result_proc.returncode != 0:
                logger.error(f"[AI_FILTER] Claude CLI failed with return code {result_proc.returncode}")
                return results
            
            try:
                # Parse JSON output (same as in test script)
                parsed_result = self._parse_claude_json_response(result_proc.stdout)
                
                if parsed_result and 'relevant_indices' in parsed_result:
                    relevant_indices = parsed_result['relevant_indices']
                    
                    # Filter results based on Claude's evaluation
                    filtered_results = []
                    for idx in relevant_indices:
                        if 1 <= idx <= len(results):  # Claude uses 1-based indexing
                            filtered_results.append(results[idx - 1])  # Convert to 0-based
                    
                    logger.info(f"[AI_FILTER] Filtered {len(results)} results down to {len(filtered_results)} relevant results")
                    self._log_to_file("PARSED_RESULT", json.dumps(parsed_result, indent=2), query)
                    return filtered_results
                else:
                    logger.warning("[AI_FILTER] No relevant_indices in Claude response, returning original results")
                    return results
                    
            except Exception as e:
                logger.error(f"[AI_FILTER] Error parsing Claude response: {e}, returning original results")
                self._log_to_file("ERROR", f"Parse error: {e}", query)
                return results
                
        except Exception as e:
            logger.error(f"[AI_FILTER] Unexpected error: {e}, returning original results")
            self._log_to_file("ERROR", f"Unexpected error: {e}", query)
            return results
    
    def _parse_claude_json_response(self, stdout: str) -> Optional[Dict[str, Any]]:
        """
        Parse Claude's JSON output format response.
        Same logic as in test_ai_filter_quality.py
        """
        try:
            # The output is a JSON object with a "result" field
            output_data = json.loads(stdout)
            
            # Extract the actual response from the result field
            if isinstance(output_data, dict) and 'result' in output_data:
                result_text = output_data['result']
                
                # Find JSON in the result text (Claude includes it in markdown code block)
                json_start = result_text.find('```json')
                if json_start >= 0:
                    # Extract content between ```json and ```
                    json_start = result_text.find('{', json_start)
                    json_end = result_text.find('```', json_start)
                    if json_end > json_start:
                        json_text = result_text[json_start:json_end].strip()
                        return json.loads(json_text)
                else:
                    # Try to find raw JSON without markdown
                    json_start = result_text.find('{')
                    json_end = result_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_text = result_text[json_start:json_end]
                        return json.loads(json_text)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[AI_FILTER] Failed to parse JSON from Claude response: {e}")
            return None
    
    def _build_evaluation_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Build the evaluation prompt for Claude using the loaded template."""
        # Import env_config to get preview lines settings
        from .. import env_config
        ai_filter_lines = env_config.get_preview_lines_ai_filter()
        
        # Format results for Claude evaluation
        formatted_results = []
        formatted_results.append(f"Found {len(results)} files matching '{query}'")
        formatted_results.append("")
        
        for i, result in enumerate(results, 1):
            formatted_results.append(f"=== [{i}] {result['path']} ===")
            # Truncate content for AI filter
            content = result['content']
            if ai_filter_lines != -1:
                lines = content.split('\n')
                if len(lines) > ai_filter_lines:
                    content = '\n'.join(lines[:ai_filter_lines]) + '\n...'
            formatted_results.append(content)
            formatted_results.append("")
        
        formatted_results.append(f"--- Total: {len(results)} files found ---")
        results_text = "\n".join(formatted_results)
        
        # Use the loaded prompt template with placeholders
        prompt = self.prompt_template.format(
            query=query,
            num_results=len(results),
            results_text=results_text
        )
        
        return prompt
    
    def _log_to_file(self, log_type: str, content: str, query: str):
        """Log to aifilter.log for debugging purposes."""
        log_dir = Path(__file__).parent.parent.parent / "Logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "aifilter.log"
        
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write("\n" + "="*80 + "\n")
                log_file.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
                log_file.write(f"MODEL: {self.model}\n")
                log_file.write(f"QUERY: {query[:100]}...\n" if len(query) > 100 else f"QUERY: {query}\n")
                log_file.write(f"TYPE: {log_type}\n")
                log_file.write("="*80 + "\n")
                log_file.write(content)
                log_file.write("\n")
        except Exception as e:
            logger.warning(f"[AI_FILTER] Failed to write to log file: {e}")