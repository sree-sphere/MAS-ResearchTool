"""
Usage of Multi-Agent System (MAS) with LangGraph and OpenAI GPT
This script demonstrates a multi-agent system that can perform research, write articles, and utilize various tools in a coordinated manner.
It includes mock data for demonstration purposes and can also run with real OpenAI API integration."""

import os
from multi_agent_system import MultiAgentSystem, ResearchResult, ArticleContent
from dotenv import load_dotenv
load_dotenv()

def run_demo_with_mock_data():
    """Run a demonstration using mock data (no API key required)"""
    print("Running Multi-Agent System Demo with Mock Data")
    print("="*60)
    
    # Create mock research result
    mock_research = ResearchResult(
        topic="Artificial Intelligence in Healthcare",
        key_findings=[
            "AI is revolutionizing medical diagnosis with 95% accuracy rates",
            "Machine learning algorithms can predict patient outcomes",
            "Automated drug discovery is reducing development time by 40%",
            "AI-powered robotic surgery is improving precision"
        ],
        sources=[
            "Medical AI Journal 2024",
            "Healthcare Technology Review",
            "Nature Medicine AI Research"
        ],
        summary="AI technologies are transforming healthcare through improved diagnostics, predictive analytics, and automated drug discovery, leading to better patient outcomes and reduced costs."
    )
    
    # Create mock article based on research
    mock_article = ArticleContent(
        title="The Revolutionary Impact of AI in Modern Healthcare",
        introduction="Artificial Intelligence is reshaping the healthcare landscape, offering unprecedented opportunities to improve patient care, reduce costs, and accelerate medical breakthroughs.",
        main_content=[
            "Diagnostic Revolution: AI systems now achieve 95% accuracy in medical image analysis, surpassing human radiologists in detecting early-stage cancers and complex conditions.",
            "Predictive Healthcare: Machine learning algorithms analyze patient data to predict health outcomes, enabling preventive care and personalized treatment plans.",
            "Drug Discovery Acceleration: AI-powered research is reducing drug development timelines from decades to years, with automated screening of millions of compounds.",
            "Surgical Precision: Robotic surgery assisted by AI provides unprecedented precision, reducing recovery times and improving surgical outcomes."
        ],
        conclusion="As AI continues to evolve, its integration into healthcare promises a future where medical care is more accurate, accessible, and personalized than ever before.",
        word_count=156
    )
    
    print(f"Research Phase Complete:")
    print(f"   Topic: {mock_research.topic}")
    print(f"   Key Findings: {len(mock_research.key_findings)} insights discovered")
    print(f"   Sources: {len(mock_research.sources)} references analyzed")
    print()
    
    print(f"Writing Phase Complete:")
    print(f"   Title: {mock_article.title}")
    print(f"   Sections: {len(mock_article.main_content)} main content areas")
    print(f"   Word Count: {mock_article.word_count} words")
    print()
    
    print("Article Preview:")
    print("-" * 40)
    print(f"Title: {mock_article.title}")
    print()
    print(f"Introduction: {mock_article.introduction}")
    print()
    print("Main Content:")
    for i, section in enumerate(mock_article.main_content, 1):
        print(f"{i}. {section[:100]}...")
    print()
    print(f"Conclusion: {mock_article.conclusion}")
    
    return mock_research, mock_article

def run_with_openai_api():
    """Run the actual multi-agent system with OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables")
        print("To use the real system:")
        print("   1. Get an OpenAI API key from https://platform.openai.com/")
        print("   2. Create a .env file with: OPENAI_API_KEY=your_key_here")
        print("   3. Run this script again")
        return None
    
    print("Running Multi-Agent System with OpenAI API")
    print("="*50)
    
    try:
        # Initializing system
        system = MultiAgentSystem(api_key)
        
        # Testing different topics
        topics = [
            "climate change solutions",
            "space exploration technologies", 
            "renewable energy advances"
        ]
        
        for topic in topics:
            print(f"\n🎯 Processing topic: {topic}")
            result = system.run_workflow(topic)
            
            if result and result.get("article_content"):
                article = ArticleContent(**result["article_content"])
                print(f"   ✅ Created: '{article.title}' ({article.word_count} words)")
            else:
                print(f"   ❌ Failed to process {topic}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running system: {e}")
        return None

def demonstrate_tools():
    """Demonstrate the individual tools"""
    print("\nTool Demonstration")
    print("="*30)
    
    from multi_agent_system import web_search, fact_checker, word_counter
    
    # web search
    print("1. Web Search Tool:")
    search_result = web_search.invoke({"query": "artificial intelligence"})
    print(f"   Query: 'artificial intelligence'")
    print(f"   Result: {search_result[:100]}...")
    
    # fact checker
    print("\n2. Fact Checker Tool:")
    fact_result = fact_checker.invoke({"claim": "AI will revolutionize healthcare"})
    print(f"   Claim: 'AI will revolutionize healthcare'")
    print(f"   Result: {fact_result}")
    
    # word counter
    print("\n3. Word Counter Tool:")
    text = "This is a sample text for word counting demonstration."
    word_count = word_counter.invoke({"text": text})
    print(f"   Text: '{text}'")
    print(f"   Word Count: {word_count}")

def main():
    """Main demonstration function"""
    print("Multi-Agent System Demonstration")
    print("=" * 60)
    print()
    
    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key detected - running full system")
        result = run_with_openai_api()
    else:
        print("No API key - running mock demonstration")
        result = run_demo_with_mock_data()
    
    # Demonstrate individual tools
    demonstrate_tools()
    
    print("\n" + "="*60)
    print("Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("   ✅ Multi-agent coordination with LangGraph")
    print("   ✅ Structured data with Pydantic models")
    print("   ✅ Tool calling capabilities")
    print("   ✅ Agent state management")
    print("   ✅ Workflow orchestration")
    
    print("\nTechnologies Used:")
    print("   • LangChain: Agent framework and tool integration")
    print("   • LangGraph: Workflow orchestration and state management") 
    print("   • Pydantic: Data validation and structured models")
    print("   • OpenAI GPT: Language model for intelligent responses")
    
    return result

if __name__ == "__main__":
    main()