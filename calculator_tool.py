import numexpr as ne
import re
from typing import Dict, Any, Optional

class CalculatorTool:
    """Tool for evaluating mathematical expressions"""
    
    name = "calculator"
    description = "Use this tool for mathematical calculations. Input should be a mathematical expression."
    
    def __init__(self):
        """Initialize the calculator tool"""
        pass
    
    def extract_expression(self, query: str) -> str:
        """
        Extract a mathematical expression from a natural language query
        
        Args:
            query: User query like "Calculate 2 + 2" or "What is 5 * 10?"
            
        Returns:
            Extracted mathematical expression
        """
        # Remove common phrases
        cleaned = query.lower()
        phrases_to_remove = [
            "calculate", "compute", "evaluate", "solve", "what is", 
            "result of", "value of", "equals", "equal to"
        ]
        
        for phrase in phrases_to_remove:
            cleaned = cleaned.replace(phrase, "")
        
        # Handle natural language mathematical operations
        cleaned = cleaned.replace("multiplied to", "*")
        cleaned = cleaned.replace("multiplied by", "*")
        cleaned = cleaned.replace("times", "*")
        cleaned = cleaned.replace("divided by", "/")
        cleaned = cleaned.replace("plus", "+")
        cleaned = cleaned.replace("minus", "-")
        cleaned = cleaned.replace("added to", "+")
        cleaned = cleaned.replace("subtracted from", "-")
        
        # Remove extra spaces and punctuation except mathematical symbols
        cleaned = re.sub(r'[^\d+\-*/().^%\s]', '', cleaned).strip()
        
        # Ensure there are spaces around operators for better parsing
        cleaned = re.sub(r'(\d+)([+\-*/])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([+\-*/])(\d+)', r'\1 \2', cleaned)
        
        return cleaned
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Process a calculation request
        
        Args:
            query: User query containing a mathematical expression
            
        Returns:
            Dictionary with tool execution results
        """
        try:
            # Extract the expression
            expression = self.extract_expression(query)
            
            # Safety check
            if not expression or len(expression) > 200:
                return {
                    "tool": self.name,
                    "input": query,
                    "expression": expression,
                    "output": "Invalid or too complex expression",
                    "success": False
                }
            
            # Evaluate expression
            result = ne.evaluate(expression)
            
            return {
                "tool": self.name,
                "input": query,
                "expression": expression,
                "output": f"The result of {expression} is {result}",
                "result": float(result),
                "success": True
            }
        except Exception as e:
            return {
                "tool": self.name,
                "input": query,
                "output": f"Error evaluating expression: {str(e)}",
                "success": False
            } 