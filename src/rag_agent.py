"""
RAG Agent - Handles ChromaDB storage and retrieval
"""

import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.log import logger


class RAGAgent:
    """RAG Agent for ChromaDB operations and content generation"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="research_results",
            metadata={"hnsw:space": "cosine"}
        )
        self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0.3)
    
    async def retrieve_and_generate(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve relevant documents and generate response"""
        try:
            # Retrieve similar documents
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"][0]:
                return {
                    "generated_content": [],
                    "retrieved_docs": [],
                    "confidence": 0.0
                }
            
            # Prepare context from retrieved documents
            context_docs = []
            for doc, metadata, distance in zip(
                results["documents"][0], 
                results["metadatas"][0], 
                results["distances"][0]
            ):
                context_docs.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance  # Convert distance to similarity
                })
            
            # Generate response using retrieved context
            generated_content = await self._generate_from_context(query, context_docs)
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(doc["similarity"] for doc in context_docs) / len(context_docs)
            
            return {
                "generated_content": generated_content,
                "retrieved_docs": context_docs,
                "confidence": avg_similarity
            }
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return {
                "generated_content": [],
                "retrieved_docs": [],
                "confidence": 0.0
            }
    
    async def _generate_from_context(self, query: str, context_docs: List[Dict]) -> List[Dict]:
        """Generate content from retrieved context"""
        # Combine context
        context_text = "\n\n".join([
            f"Source: {doc['metadata'].get('title', 'Unknown')}\n{doc['content'][:800]}"
            for doc in context_docs
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """Based on the retrieved context, provide a comprehensive answer to the user's query.
            
            Query: {query}
            
            Context:
            {context}
            
            Requirements:
            - Use information from the provided context
            - Provide accurate and relevant information
            - Maintain professional tone
            - If context is insufficient, acknowledge limitations
            
            Response:"""
        )
        
        try:
            response = await self.llm.ainvoke(
                prompt.format_messages(query=query, context=context_text[:4000])
            )
            
            return [{
                "content_type": "rag_response",
                "title": f"Response to: {query[:100]}...",
                "content": response.content,
                "summary": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                "word_count": len(response.content.split()),
                "sources_used": [doc["metadata"].get("title", "Unknown") for doc in context_docs]
            }]
            
        except Exception as e:
            logger.error(f"Content generation from context failed: {e}")
            return []
    
    async def store_pipeline_results(self, query: str, pipeline_result: Dict) -> str:
        """Store full pipeline results in ChromaDB"""
        try:
            chroma_id = str(uuid.uuid4())
            
            # Prepare documents from generated content
            documents = []
            metadatas = []
            ids = []
            
            for i, content in enumerate(pipeline_result.get("generated_content", [])):
                doc_id = f"{chroma_id}_{i}"
                
                documents.append(content["content"])
                metadatas.append({
                    "query": query,
                    "content_type": content["content_type"],
                    "title": content["title"],
                    "word_count": content["word_count"],
                    "created_at": datetime.now().isoformat(),
                    "source": "pipeline",
                    "pipeline_id": pipeline_result.get("pipeline_id", "unknown"),
                    "chroma_group_id": chroma_id
                })
                ids.append(doc_id)
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Stored {len(documents)} pipeline documents with ID: {chroma_id}")
            
            return chroma_id
            
        except Exception as e:
            logger.error(f"Failed to store pipeline results: {e}")
            return str(uuid.uuid4())  # Return dummy ID
    
    async def store_rag_results(self, query: str, rag_result: Dict) -> str:
        """Store RAG-generated results in ChromaDB"""
        try:
            chroma_id = str(uuid.uuid4())
            
            if rag_result.get("generated_content"):
                content = rag_result["generated_content"][0]
                
                self.collection.add(
                    documents=[content["content"]],
                    metadatas=[{
                        "query": query,
                        "content_type": content["content_type"],
                        "title": content["title"],
                        "word_count": content["word_count"],
                        "created_at": datetime.now().isoformat(),
                        "source": "rag",
                        "confidence": rag_result.get("confidence", 0.0),
                        "chroma_group_id": chroma_id
                    }],
                    ids=[chroma_id]
                )
                
                logger.info(f"Stored RAG result with ID: {chroma_id}")
            
            return chroma_id
            
        except Exception as e:
            logger.error(f"Failed to store RAG results: {e}")
            return str(uuid.uuid4())
    
    def get_collection_stats(self) -> Dict:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_documents": 0, "collection_name": "unknown"}
    
    def cleanup_old_documents(self, days_old: int = 7):
        """Clean up documents older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Query all documents
            all_docs = self.collection.get(include=["metadatas"])
            
            old_ids = []
            for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
                created_at = metadata.get("created_at")
                if created_at:
                    doc_date = datetime.fromisoformat(created_at)
                    if doc_date < cutoff_date:
                        old_ids.append(doc_id)
            
            if old_ids:
                self.collection.delete(ids=old_ids)
                logger.info(f"Cleaned up {len(old_ids)} old documents")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")