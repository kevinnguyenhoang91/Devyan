from crewai import Agent
from textwrap import dedent
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI


class CustomAgents:
    def __init__(self):
        # self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        self.Ollama = Ollama(model="deepseek-coder-v2", temperature=0.7)

    def architect_agent(self, tools):
        return Agent(
            role="Software Architect",
            backstory=dedent(f"""\
            Your expertise extends beyond mere system design; you are adept at leveraging cutting-edge tools and technologies to automate workflows, optimize performance, and ensure scalability. Your methodical approach to problem-solving, combined with a deep understanding of both development and operational challenges, enables you to create robust, efficient, and reliable systems. You thrive in collaborative environments where you can mentor teams, share best practices, and continuously refine processes to achieve higher levels of productivity and innovation. Your commitment to continuous improvement and your proactive stance on addressing potential issues before they arise make you an invaluable asset to any organization."""),
            goal=dedent(f"""\
            Provide a high-level solution overview for a given problem, with a detailed breakdown of the major components and the solution. This should include researching new information and best practices, ensuring the approach leverages the latest technologies and methodologies. The overview should be comprehensive yet clear, allowing stakeholders to understand the scope, benefits, and implementation steps involved. Additionally, identify potential challenges and propose mitigation strategies to ensure the solution is robust and scalable."""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
        )

    def programmer_agent(self, tools):
        return Agent(
            role="Software Programmer",
            backstory=dedent(f"""\
            You havea keen eye for detail and a knack for translating high-level design solutions into robust,
            efficient code."""),
            goal=dedent(f"""Implement the solution provided by the architect"""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
        )

    def tester_agent(self, tools):
        return Agent(
            role="Software Tester",
            backstory=dedent(f"""\
            Your passion for quality ensures that every piece of code meets the highest
            standards through rigorous testing."""),
            goal = dedent("""\
            Write and run test cases for the code implemented by the programmer"""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
        )

    def reviewer_agent(self, tools):
        return Agent(
            role="Software Reviewer",
            backstory=dedent("""\
            With a critical eye, you review each step of the development process, ensuring quality and consistency."""),
            goal=dedent("""\
            Review the work of each agent at each step"""),
            tools=tools,
            allow_delegation=False,
            verbose=True,
            llm=self.Ollama,
        )
