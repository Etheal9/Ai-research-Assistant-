
import gradio as gr
import os
from main import query_router, handle_meta_query, rag_chain

def research_assistant(question):
    """
    this function is the core logic that Gradio will call.
    It takes a user question and returns the formatted answer.
    """

    # Route the question to the appropriate tool
    route = query_router(question)

    if route == 'meta_query':
        return handle_meta_query(question)
    
    elif route == 'rag_query':
        result = rag_chain.invoke(question)
        answer = result["answer"]

        # process and format the sources
        source_documents = result["documents"]
        unique_sources = {}
        for doc in source_documents:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            if source_file not in unique_sources:
                display_name = doc.metadata.get("title", source_file)
                if not display_name:
                    display_name = source_file
                unique_sources[source_file] = display_name
        
        # Only append sources if the answer seems to be using the research documents
        # (i.e., not just a casual greeting or general conversation)
        if unique_sources and any(keyword in question.lower() for keyword in 
                                ['agent', 'reinforcement', 'learning', 'multi-agent', 'algorithm', 'training', 'reward']):
            answer += "\n\n**Sources:**\n"
            for display_name in sorted(unique_sources.values()):
                answer += f"- {display_name}\n"

        return answer

# Build the Gradio Interface
iface = gr.Interface(
    fn=research_assistant,
    inputs=gr.Textbox(lines=3, label="Your Question", placeholder="Say hi, ask about Multi-Agent AI Systems, or anything else!"),
    outputs=gr.Markdown(label="Assistant's Response"),
    title="AI Research Assistant",
    description="I'm a friendly AI assistant with expertise in Multi-Agent AI Systems. I can chat about general topics and provide detailed answers about reinforcement learning, multi-agent systems, and generative agents based on research papers."
)

if __name__ == "__main__":
    iface.launch()
