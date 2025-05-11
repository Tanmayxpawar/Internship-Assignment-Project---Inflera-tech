import sys
import pysqlite3

# Monkey-patch stdlib sqlite3 to use the bundled, modern SQLite
sys.modules['sqlite3'] = pysqlite3
import os
from typing import List, Dict, Any, Optional, Union, ClassVar
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.agents.agent import AgentOutputParser
from document_processor import DocumentProcessor
from calculator_tool import CalculatorTool
import re
import json
from dotenv import load_dotenv
from pydantic import Field

# Load environment variables
load_dotenv()

# Get API key and verify it exists
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file.")

class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        # Parse out the action and action input
        action_match = re.search(r"Action: (.*?)[\n]*Action Input: (.*)", text, re.DOTALL)
        if not action_match:
            return AgentFinish(
                return_values={"output": "I could not determine the next action."},
                log=text,
            )
        
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)

class AgentPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the agent."""
    
    # Use ClassVar annotation for class variables that shouldn't be fields
    template: ClassVar[str] = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    {agent_scratchpad}"""
    
    # Properly annotate input_variables
    input_variables: List[str] = Field(default=["tools", "tool_names", "input", "agent_scratchpad"])

    def format(self, **kwargs) -> str:
        """Format the prompt template."""
        # Handle agent_scratchpad specially
        if "agent_scratchpad" in kwargs:
            intermediate_steps = kwargs.get("agent_scratchpad", [])
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += f"\nAction: {action}\nObservation: {observation}\nThought: "
            kwargs["agent_scratchpad"] = thoughts
        
        # Format tools list for prompt
        if "tools" in kwargs:
            tools = kwargs["tools"]
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
            tool_strings = []
            for tool in tools:
                tool_strings.append(f"{tool.name}: {tool.description}")
            kwargs["tools"] = "\n".join(tool_strings)
        
        return self.template.format(**kwargs)

class KnowledgeAssistant:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the knowledge assistant with document processing and LLM capabilities.
        
        Args:
            model_name (str): Name of the Groq model to use
        """
        self.document_processor = DocumentProcessor()
        
        # Use properly configured LLM with explicit API key
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.7,
            groq_api_key=api_key,  # Explicitly passing API key
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.calculator = CalculatorTool()
        
        # Initialize tools
        self.tools = [
            Tool(
                name="DocumentSearch",
                func=self._search_documents,
                description="Search for relevant information in the document collection"
            ),
            Tool(
                name="Calculator",
                func=self._calculate,
                description="Perform mathematical calculations"
            ),
            Tool(
                name="Dictionary",
                func=self._define_word,
                description="Look up word definitions"
            )
        ]
        
        # Initialize agent
        self.agent = self._create_agent()
        
    def _create_agent(self) -> AgentExecutor:
        """Create and configure the agent with tools and memory."""
        prompt = AgentPromptTemplate()
        output_parser = CustomOutputParser()
        
        # Create LLM chain with proper input variables
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        # Create agent with proper configuration
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in self.tools],
            stop=["\nObservation:"],
            memory=self.memory,
            output_parser=output_parser
        )
        
        # Create executor with proper configuration
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _search_documents(self, query: str) -> str:
        """Search documents using the document processor."""
        try:
            results = self.document_processor.query_documents(query, n_results=3)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def _calculate(self, query: str) -> str:
        """Process calculation requests using the calculator tool."""
        try:
            result = self.calculator.run(query)
            return result["output"] if isinstance(result, dict) and "output" in result else str(result)
        except Exception as e:
            return f"Error performing calculation: {str(e)}"
    
    def _define_word(self, word: str) -> str:
        """Look up word definitions using Groq."""
        try:
            prompt = f"Define the word '{word}' in a clear and concise way."
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Error looking up definition: {str(e)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the system.
        
        Args:
            query (str): The user's query
            
        Returns:
            Dict[str, Any]: Results including the chosen tool, retrieved context, and final answer
        """
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(query)
            
            # First check if it's a calculation request
            if any(keyword in query.lower() for keyword in ["calculate", "compute", "math", "+", "-", "*", "/", "multiplied", "divided", "plus", "minus", "times"]):
                result = self._calculate(query)
                # Add assistant response to memory
                self.memory.chat_memory.add_ai_message(result)
                return {
                    "tool_used": "Calculator",
                    "answer": result,
                    "context": None
                }
            
            # Then check if it's a dictionary request
            if any(keyword in query.lower() for keyword in ["define", "meaning of", "what is"]):
                result = self._define_word(query)
                # Add assistant response to memory
                self.memory.chat_memory.add_ai_message(result)
                return {
                    "tool_used": "Dictionary",
                    "answer": result,
                    "context": None
                }
            
            # If not a specific tool request, search documents
            context = self._search_documents(query)
            
            # If we have relevant context, use it
            if context and context != "[]":
                return {
                    "tool_used": "DocumentSearch",
                    "answer": None,  # Let the chain handle the answer generation
                    "context": context
                }
            
            # If no specific tool matches and no context found
            return {
                "tool_used": "DirectLLM",
                "answer": None,  # Let the chain handle the answer generation
                "context": None
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            # Add error message to memory
            self.memory.chat_memory.add_ai_message(error_msg)
            return {
                "tool_used": "Error",
                "answer": error_msg,
                "context": None
            }
    
    def ingest_documents(self, directory_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Ingest documents into the system.
        
        Args:
            directory_path (str): Path to directory containing documents
            metadata (Dict[str, Any], optional): Additional metadata for documents
        """
        try:
            self.document_processor.process_directory(directory_path, metadata)
        except Exception as e:
            print(f"Error ingesting documents: {str(e)}")

def main():
    """Main function to run the assistant."""
    try:
        # Initialize the assistant
        assistant = KnowledgeAssistant()
        print("Knowledge Assistant initialized successfully!")
        
        # Example query
        query = "What is the capital of France?"
        print(f"\nProcessing query: '{query}'")
        
        result = assistant.process_query(query)
        print("\nResult:")
        print(f"Tool used: {result['tool_used']}")
        if result['context']:
            print(f"Context: {result['context']}")
        print(f"Answer: {result['answer']}")
        
    except Exception as e:
        print(f"Error running the assistant: {str(e)}")

if __name__ == "__main__":
    main()