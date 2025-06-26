"""
Multi-Agent System with Tool Calling
A simple demonstration of LangChain + LangGraph + Pydantic for multi-agent workflows
"""

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

# Pydantic Models for structured data
class ResearchResult(BaseModel):
    """Structured research result"""
    topic: str
    key_findings: List[str] = Field(description="Key findings from research")
    sources: List[str] = Field(description="Sources of information")
    summary: str = Field(description="Brief summary of findings")

class ArticleContent(BaseModel):
    """Structured article content"""
    title: str
    introduction: str
    main_content: List[str] = Field(description="Main content sections")
    conclusion: str
    word_count: int

class AgentState(BaseModel):
    """State shared between agents"""
    messages: List[BaseMessage] = Field(default_factory=list)
    research_results: Optional[ResearchResult] = None
    article_content: Optional[ArticleContent] = None
    task: str = ""
    current_step: str = "start"

# Mock Tools (In production, these would connect to real APIs)
@tool
def web_search(query: str) -> str:
    """Mock web search tool that simulates searching the internet"""
    # Mock search results - in real implementation, this would call actual search APIs
    mock_results = {
        "artificial intelligence": [
            "AI is transforming industries worldwide",
            "Machine learning advances in 2024",
            "Ethics in AI development becoming crucial"
        ],
        "climate change": [
            "Global temperatures rising at unprecedented rates",
            "Renewable energy adoption accelerating",
            "Carbon capture technologies showing promise"
        ],
        "space exploration": [
            "Mars rover missions providing new insights",
            "Private space companies revolutionizing access",
            "International space station research expanding"
        ]
    }
    
    # Simple keyword matching for demonstration
    for topic, results in mock_results.items():
        if topic.lower() in query.lower():
            return f"Search results for '{query}':\n" + "\n".join([f"- {result}" for result in results])
    
    return f"Search results for '{query}':\n- General information about {query}\n- Recent developments in {query}\n- {query} impact on society"

@tool
def fact_checker(claim: str) -> str:
    """Mock fact-checking tool"""
    return f"Fact-check result: The claim '{claim}' appears to be generally accurate based on available sources."

@tool
def word_counter(text: str) -> int:
    """Tool to count words in text"""
    return len(text.split())

# Agent Classes
class ResearchAgent:
    """Agent responsible for gathering and analyzing information"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [web_search, fact_checker]
    
    def research_topic(self, topic: str) -> ResearchResult:
        """Conduct research on a given topic"""
        # Use the web search tool
        search_results = web_search.invoke({"query": topic})
        
        # Create a research prompt
        prompt = f"""
        You are a research expert. Based on the following search results about {topic}, 
        provide a structured analysis:
        
        Search Results:
        {search_results}
        
        Please provide:
        1. Key findings (3-5 main points)
        2. Sources (even if mock)
        3. A brief summary
        
        Format your response as a JSON object with keys: key_findings, sources, summary
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse the response (simplified - in production would use better parsing)
        try:
            # Extract JSON from response if possible
            content = response.content
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_content = content[start:end]
                data = json.loads(json_content)
            else:
                # Fallback if no JSON
                data = {
                    "key_findings": ["Research finding 1", "Research finding 2", "Research finding 3"],
                    "sources": ["Source 1", "Source 2"],
                    "summary": f"Research summary about {topic}"
                }
        except:
            # Fallback data
            data = {
                "key_findings": [f"Key insight about {topic}", f"Important trend in {topic}", f"Future implications of {topic}"],
                "sources": ["Mock Source 1", "Mock Source 2"],
                "summary": f"Comprehensive research summary about {topic} based on available information."
            }
        
        return ResearchResult(
            topic=topic,
            key_findings=data.get("key_findings", []),
            sources=data.get("sources", []),
            summary=data.get("summary", "")
        )

class WritingAgent:
    """Agent responsible for content creation"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [word_counter]
    
    def create_article(self, research: ResearchResult) -> ArticleContent:
        """Create an article based on research results"""
        prompt = f"""
        You are a professional writer. Based on the following research about {research.topic}, 
        create a well-structured article.
        
        Research Summary: {research.summary}
        Key Findings: {', '.join(research.key_findings)}
        
        Please write an article with:
        1. An engaging title
        2. A compelling introduction (2-3 sentences)
        3. Main content (3-4 key sections)
        4. A strong conclusion
        
        Format your response as JSON with keys: title, introduction, main_content (array), conclusion
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse the response
        try:
            content = response.content
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_content = content[start:end]
                data = json.loads(json_content)
            else:
                # Fallback structure
                data = {
                    "title": f"Understanding {research.topic}",
                    "introduction": f"In recent years, {research.topic} has become increasingly important.",
                    "main_content": research.key_findings,
                    "conclusion": f"The future of {research.topic} looks promising with continued developments."
                }
        except:
            # Fallback data
            data = {
                "title": f"Comprehensive Guide to {research.topic}",
                "introduction": f"{research.topic} represents one of the most significant developments in our modern world.",
                "main_content": research.key_findings,
                "conclusion": f"As we move forward, {research.topic} will continue to shape our future."
            }
        
        # Calculate word count
        full_text = f"{data['title']} {data['introduction']} {' '.join(data.get('main_content', []))} {data['conclusion']}"
        word_count = word_counter.invoke({"text": full_text})
        
        return ArticleContent(
            title=data.get("title", ""),
            introduction=data.get("introduction", ""),
            main_content=data.get("main_content", []),
            conclusion=data.get("conclusion", ""),
            word_count=word_count
        )

# Multi-Agent Workflow using LangGraph
class MultiAgentSystem:
    """Orchestrates multiple agents using LangGraph"""
    
    def __init__(self, openai_api_key: str):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.7
        )
        
        # Initialize agents
        self.research_agent = ResearchAgent(self.llm)
        self.writing_agent = WritingAgent(self.llm)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        def research_step(state: Dict[str, Any]) -> Dict[str, Any]:
            """Research step in the workflow"""
            print(f"Starting research on: {state['task']}")
            
            research_results = self.research_agent.research_topic(state["task"])
            
            return {
                **state,
                "research_results": research_results.dict(),
                "current_step": "research_complete",
                "messages": state["messages"] + [
                    AIMessage(content=f"Research completed on {state['task']}. Found {len(research_results.key_findings)} key findings.")
                ]
            }
        
        def writing_step(state: Dict[str, Any]) -> Dict[str, Any]:
            """Writing step in the workflow"""
            print(f"✍️  Creating article based on research...")
            
            # Convert dict back to ResearchResult
            research_data = state["research_results"]
            research_result = ResearchResult(**research_data)
            
            article_content = self.writing_agent.create_article(research_result)
            
            return {
                **state,
                "article_content": article_content.dict(),
                "current_step": "writing_complete",
                "messages": state["messages"] + [
                    AIMessage(content=f"Article created: '{article_content.title}' ({article_content.word_count} words)")
                ]
            }
        
        def should_continue(state: Dict[str, Any]) -> str:
            """Determine next step in workflow"""
            current_step = state.get("current_step", "start")
            
            if current_step == "start":
                return "research"
            elif current_step == "research_complete":
                return "writing"
            else:
                return END
        
        # Build the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("research", research_step)
        workflow.add_node("writing", writing_step)
        
        # Add edges
        workflow.set_conditional_entry_point(should_continue)
        workflow.add_conditional_edges("research", should_continue)
        workflow.add_conditional_edges("writing", should_continue)
        
        # Compile the workflow
        return workflow.compile(checkpointer=MemorySaver())
    
    def run_workflow(self, task: str) -> Dict[str, Any]:
        """Run the complete multi-agent workflow"""
        print(f"🚀 Starting multi-agent workflow for task: {task}")
        
        # Initial state
        initial_state = {
            "task": task,
            "current_step": "start",
            "messages": [HumanMessage(content=f"Task: {task}")],
            "research_results": None,
            "article_content": None
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": "demo_thread"}}
        final_state = self.workflow.invoke(initial_state, config)
        
        print("Workflow completed!")
        return final_state

def demonstrate_system():
    """Demonstrate the multi-agent system"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        print("For demo purposes, using mock responses...")
        api_key = "demo-key"
    
    try:
        # Initialize the multi-agent system
        system = MultiAgentSystem(api_key)
        
        # Run a sample task
        task = "artificial intelligence"
        result = system.run_workflow(task)
        
        # Display results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        if result.get("research_results"):
            research = ResearchResult(**result["research_results"])
            print(f"\nResearch Results for '{research.topic}':")
            print(f"Summary: {research.summary}")
            print(f"Key Findings: {', '.join(research.key_findings[:2])}...")
        
        if result.get("article_content"):
            article = ArticleContent(**result["article_content"])
            print(f"\nArticle Created:")
            print(f"Title: {article.title}")
            print(f"Word Count: {article.word_count}")
            print(f"Introduction: {article.introduction[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        print("This is likely due to missing OpenAI API key")
        return None

if __name__ == "__main__":
    demonstrate_system()