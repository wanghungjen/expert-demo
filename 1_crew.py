from crewai import Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="crewai-llama3",
    base_url="http://localhost:11434/v1"
)

# --- Tools ---
# PDF SOURCE: https://www.gpinspect.com/wp-content/uploads/2021/03/sample-home-report-inspection.pdf
pdf_search_tool = PDFSearchTool(
    pdf="./attention_paper.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3")),
        embedder=dict(provider="ollama", config=dict(model="all-minilm")),
    ),
)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The research agent is adept at searching and 
        extracting data from documents, ensuring accurate and prompt responses.
        """
    ),
    tools=[pdf_search_tool],
    llm = llm
)

expert_agent = Agent(
    role="Expert Agent",
    goal="Critique on the research agent's findings and extract relevant information",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The expert agent is an expert in the field of computer science, especially aritifical intellingence
        and is able to provide technical feedback on the provided information.
        """
    ),
    tools=[],
    llm = llm
)

professional_writer_agent = Agent(
    role="Professional Writer",
    goal="Write professional paragraphs based on the research agent's findings and expert agent's critiques",
    allow_delegation=False,
    verbose=True,
    backstory=(
        """
        The professional writer agent has excellent writing skills and is able to craft 
        clear and concise paragraphs based on the provided information.
        """
    ),
    tools=[],
    llm = llm
)


# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the user's question based on the given research PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the research PDF.

        Here is the user's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the user's questions based on 
        the content of the given research PDF.
        """,
    tools=[pdf_search_tool],
    agent=research_agent,
)

critique_task = Task(
    description = (
        """
        - Critique verbosely on the research agent's findings and provide valuable insights
        - The responses should clearly explain whether the points are related to the user's question

        Here is the user's question:
        {customer_question}
        """
    ),
    expected_output = """
        Provide clear and accurate critiques to each answer provided by the research agent
    """,
    tools=[],
    agent = expert_agent,
)

summarizing_task = Task(
    description=(
        """
        - Write one paragraph to answer the user based on the research agent's findings and expert agent's critiques.
        - The paragraph should clearly respond to the user's question given at the start

        Here is the user's question:
        {customer_question}
        """
    ),
    expected_output="""
        Write one clear and concise paragraph that can be sent to the user to address the
        question that they had.
        """,
    tools=[],
    agent=professional_writer_agent,
)

# --- Crew ---
crew = Crew(
    agents=[research_agent, expert_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, critique_task, summarizing_task],
    process=Process.sequential,
)

customer_question = input(
    "How can I help you today?\n"
)
result = crew.kickoff(inputs={"customer_question": customer_question})
print()
print("------")
print(result)
print("------")
