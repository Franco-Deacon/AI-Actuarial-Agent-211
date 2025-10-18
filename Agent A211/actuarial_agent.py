import chromadb
import time
import threading
import math
import os
import operator
import json
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define constants for the vector store
CHROMA_PATH = "actuarial_notes_db"
NOTES_FILE = "Notes.pdf"

# Using the provided fallback for ToolExecutor for compatibility
class ToolExecutor:
    def __init__(self, tools):
        self._tools = {tool.name: tool for tool in tools}

    def invoke(self, tool_call: dict):
        if tool_call["name"] in self._tools:
            return self._tools[tool_call["name"]].invoke(tool_call["args"])
        raise ValueError(f"Tool '{tool_call['name']}' not found.")

#___ 1. Define the Tools (annuity and interest rate formulaes) ___#
@tool
def effective_interest_from_nominal_interest(
    i_p: float, p: int
) -> float:
    """
    Relationship between the effective and nominal interest rates.
    Args:
        i_p: The nominal interest rate per annum (as a decimal, e.g., 0.06 for 6%).
        p: The number of compounding periods per year. (compunded mothly, p=12)
    Returns:
        The effective interest rate (as a decimal).
    """
    i = ((1 + i_p / p) ** p) - 1
    return i

@tool
def effective_discount_from_nominal_interest(
    d_p: float, p: int
) -> float:
    """
    Relationship between the effective and nominal discount rates.
    d = 1 - (1 - d_p/p)^p
    """
    d = 1 - (1 - d_p / p) ** p
    return d

@tool
def nominal_interest_from_effective_interest(
    i: float, p: int
) -> float:
    """
    Relationship between the effective and nominal interest rates.
    i_p = p * ((1 + i) ** (1 / p) - 1)
    """
    i_p = p * ((1 + i) ** (1 / p) - 1)
    return i_p

@tool
def nominal_discount_from_effective_discount(
    d: float, p: int
) -> float:
    """
    Relationship between the nominal discount rate and effective discount.
    d_p = p * (1 - (1 - d) ** (1 / p))
    """
    d_p = p * (1 - (1 - d) ** (1 / p))
    return d_p

@tool
def effective_interest_from_nominal_discount(
    d_p: float, p: float 
    ) -> float:
    """
    Relationship between the effective interest rate and nominal discount rate.
    i = (1 - d_p/p)^(-p) - 1
    """
    i = (1 - d_p / p) ** (-p) - 1
    return i

@tool
def nominal_discount_from_effective_interest(
    i: float, p: int
) -> float:
    """
    Relationship between the nominal discount rate and effective interest rate.
    d_p = p * (1 - (1 + i) ** (-1 / p))
    """
    d_p = p * (1 - (1 + i) ** (-1 / p))
    return d_p

@tool
def effective_interest_from_force_of_interest(
    delta: float
) -> float:
    """
    Relationship between the effective interest rate and force of interest.
    i = e^delta - 1
    """
    i = math.exp(delta) - 1
    return i

@tool
def force_of_interest_from_effective_interest(
    i: float
) -> float:
    """
    Relationship between the force of interest and effective interest rate.
    delta = ln(1 + i)
    """
    delta = math.log(1 + i)
    return delta

@tool
def real_rate_of_interest(
    i: float, j: float
) -> float:
    """
    Relationship between the nominal rate of interest, real rate of interest, and inflation rate.
    Args:
        i: the effective money rate of interest, per annum (as a decimal).
        j: the effective inflation rate, per annum (as a decimal).
    """
    i_prime = (i - j) / (1 + j)
    return i_prime

@tool
def money_rate_of_interest_from_real_inflation (
    i_prime: float, j: float
) -> float:
    """
    Relationship between the nominal rate of interest, real rate of interest, and inflation rate.
    Args:
        i_prime: the effective real rate of interest, per annum (as a decimal).
        j: the effective inflation rate, per annum (as a decimal).
    """
    i = (1 + i_prime) * (1 + j) - 1 
    return i

@tool
def present_value_arrear_annuity(
    n: float, i: float, payment: float = 1.0
) -> float:
    """
    Present value of an annuity-immediate (payments at end of period).
    """
    if i == 0:
        return n * payment
    v = 1 / (1 + i)
    a_n = (1 - v**n) / i
    return payment * a_n

@tool
def present_value_advance_annuity(
    n: float, i: float, payment: float = 1.0
) -> float:
    """
    Present value of an annuity-due (payments at beginning of period).
    """
    if i == 0:
        return n * payment
    v = 1 / (1 + i)
    d = i * v
    a_double_dot_n = (1 - v**n) / d
    return payment * a_double_dot_n

@tool
def future_value_arrear_annuity(
    n: float, i: float, payment: float = 1.0
) -> float:
    """
    Future value (accumulated value) of annuity-immediate.
    """
    if i == 0:
        return n * payment
    s_n = ((1 + i) ** n - 1) / i
    return payment * s_n

@tool
def future_value_advance_annuity(
    n: float, i: float, payment: float = 1.0
) -> float:
    """
    Future value (accumulated value) of annuity-due.
    """
    if i == 0:
        return n * payment
    v = 1 / (1 + i)
    d = i * v
    s_double_dot_n = ((1 + i) ** n - 1) / d
    return payment * s_double_dot_n

@tool
def present_value_continuous_annuity(
    n: float, delta: float, payment_per_annum: float = 1.0
) -> float:
    """
    Present value of a continuous annuity.
    """
    if delta == 0:
        return n * payment_per_annum
    v_n = math.exp(-delta * n)
    a_n_continuous = (1 - v_n) / delta
    return payment_per_annum * a_n_continuous

@tool
def future_value_continuous_annuity(
    n: float, delta: float, payment_per_annum: float = 1.0
) -> float:
    """
    Future value of a continuous annuity.
    """
    if delta == 0:
        return n * payment_per_annum
    e_delta_n = math.exp(delta * n)
    s_n_continuous = (e_delta_n - 1) / delta
    return payment_per_annum * s_n_continuous

@tool
def present_value_pthly_arrear_annuity(n: float, p: int, i: float, payment_per_annum: float = 1.0) -> float:
    """Present value of an annuity-immediate payable p-thly."""
    i_p = p * ((1 + i) ** (1 / p) - 1)
    if i_p == 0: return n * payment_per_annum
    v_n = (1 + i) ** -n
    a_n_p = (1 - v_n) / i_p
    return payment_per_annum * a_n_p

@tool
def future_value_pthly_arrear_annuity(n: float, p: int, i: float, payment_per_annum: float = 1.0) -> float:
    """Future value of an annuity-immediate payable p-thly."""
    i_p = p * ((1 + i) ** (1 / p) - 1)
    if i_p == 0: return n * payment_per_annum
    s_n_p = ((1 + i)**n - 1) / i_p
    return payment_per_annum * s_n_p

@tool
def present_value_pthly_advance_annuity(n: float, p: int, i: float, payment_per_annum: float = 1.0) -> float:
    """Present value of an annuity-due payable p-thly."""
    d = i / (1 + i)
    d_p = p * (1 - (1 - d) ** (1 / p))
    if d_p == 0: return n * payment_per_annum
    v_n = (1 + i) ** -n
    a_double_dot_n_p = (1 - v_n) / d_p
    return payment_per_annum * a_double_dot_n_p

@tool
def future_value_pthly_advance_annuity(n: float, p: int, i: float, payment_per_annum: float = 1.0) -> float:
    """Future value of an annuity-due payable p-thly."""
    d = i / (1 + i)
    d_p = p * (1 - (1 - d) ** (1 / p))
    if d_p == 0: return n * payment_per_annum
    s_double_dot_n_p = ((1 + i)**n - 1) / d_p
    return payment_per_annum * s_double_dot_n_p

@tool
def discrete_increasing_annuity_arrear(
    n:float, i:float, payment_increase:float=1.0
    ) -> float:
    """
    Present value of a discrete increasing annuity-immediate.
    """
    v = 1 / (1 + i)
    d = i * v
    a_double_dot_n = (1 - v**n) / d
    I_a_n =(a_double_dot_n - n * v ** n) / i
    return payment_increase * I_a_n

@tool
def discrete_increasing_annuity_advance(
    n:float, i:float, payment_increase:float=1.0
    ) -> float:
    """
    Present value of a discrete increasing annuity-due.
    """
    v = 1 / (1 + i)
    d = i * v
    a_double_dot_n = (1 - v**n) / d
    I_a_double_dot_n =(a_double_dot_n - n * v ** n) / d
    return payment_increase * I_a_double_dot_n

@tool
def discrete_increasing_annuity_continuous(
    n:float, delta:float, payment_increase:float=1.0
    ) -> float:
    """
    Present value of a discrete increasing annuity-continuous. Discrete increasing payments,
    compounded continuously.
    """
    i = math.exp(delta) - 1
    v = 1 / (1 + i)
    d = i * v
    a_double_dot_n = (1 - v**n) / d
    I_a_continuous_n =(a_double_dot_n - n * v ** n) / delta
    return payment_increase * I_a_continuous_n

@tool
def perpuitty_arrear(
    i: float, payment: float = 1.0
) -> float:
    """
    Present value of an annuity in arrears that pays indefinitely.
    """
    if i <= 0:
        raise ValueError("Interest rate must be positive for perpetuity calculation.")
    a_infinity = 1 / i
    return payment * a_infinity

@tool
def perpuitty_advance(
    i: float, payment: float = 1.0
) -> float:
    """
    Present value of an annuity in advance that pays indefinitely.
    """
    if i <= 0:
        raise ValueError("Interest rate must be positive for perpetuity calculation.")
    d = i / (1 + i)
    a_double_dot_infinity = 1 / d
    return payment * a_double_dot_infinity

@tool
def pv_perpetuity_from_d1(
    D1: float, g: float, i: float
) -> float:
    """
    Calculates the price of a growing perpetuity given the dividend at the END of the first year (D1).
    Use this when the problem specifies the 'first payment'.
    Formula: P = D1 / (i - g)
    Args: 
        D1: Dividend/Coupon at the end of the first year.
        g: Growth rate of the dividend (as a decimal).
        i: Required rate of return or discount rate (as a decimal).
    """
    if i <= g:
        raise ValueError("Interest rate must be greater than the growth rate.")
    return D1 / (i - g)

@tool
def pv_perpetuity_from_d0(
    D0: float, g: float, i: float
) -> float:
    """
    Calculates the price of a growing perpetuity given the dividend that was JUST PAID (D0).
    Use this when the problem specifies the 'most recent payment' or 'payment just made'.
    Formula: P = D0 * (1 + g) / (i - g)
    Args: 
        D0: Dividend that was just paid.
        g: Growth rate of the dividend (as a decimal).
        i: Required rate of return or discount rate (as a decimal).
    """
    if i <= g:
        raise ValueError("Interest rate must be greater than the growth rate.")
    return (D0 * (1 + g)) / (i - g)

@tool
def price_bond_with_tax(
    i: float, p: float, n:float, C:float, R:float, t_1:float, t_2:float
) -> float:    
    """
    Price of a bond considering taxation on coupons and redemption.
    Args:
        i: effective interest rate per annum (as a decimal).
        i_p: Nominal interest rate per annum (as a decimal).
        p: Number of compounding periods per year.
        n: Number of years to maturity.
        C: Annual coupon payment (as a decimal of nominal value).
        R: Redemption value (as a decimal of nominal value).
        t_1: Income Tax rate on coupon income (as a decimal).
        t_2: Capital Gain Tax rate on redemption income (as a decimal).
    Returns:
        The price of the bond.
    """
    i_p = p * ((1 + i) ** (1 / p) - 1)
    v = 1 / (1 + i)
    a_n_p = (1 - v**n) / i_p
    P = ((C * (1 - t_1) * a_n_p) + (R * v**n * (1 - t_2))) / (1 - t_2 * v ** n)
    return P

@tool
def loan_prospective(
    L_0: float, i: float, n: int, X: float) -> float:
    """
    The loan outstand at time n, jsut after the nth payement, is the present value
    at time n of the future repayment instalments.
    Args:
        L_0: The initial loan amount.
        i: The effective interest rate per period (as a decimal).
        n: The number of periods elapsed.
        X: The regular repayment instalment amount.
    """
    L_n_plus = L_0 * (1 + i) ** n - X * ((1 + i) ** n - 1) / i
    return L_n_plus

@tool
def course_notes_retriever(query: str) -> str:
    """
    Searches through the studen'ts uploaded notes from a persistent vector store.
    If the store doesn't exist, it will build it once from the PDF.
    """
    # The first time you run this, it will download the model (about 220MB).
    # After that, it will load it instantly from your local cache.
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(CHROMA_PATH):
        print(f"Database not found at '{CHROMA_PATH}'. Building a new one...")
        
        # Check for the source PDF before trying to build
        if not os.path.exists(NOTES_FILE):
            return f"Error: The notes file '{NOTES_FILE}' was not found. Cannot build the database."
            
        try:
            # Load and split the PDF
            loader = PyPDFLoader(NOTES_FILE)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000, 
                            chunk_overlap=400,
                            separators=["\n\n", "\n", ". ", " ", ""]    
                         )
            splits = text_splitter.split_documents(documents)
            
            # Create and persist the vector store
            Chroma.from_documents(
                documents=splits, 
                embedding=embedding_model,
                persist_directory=CHROMA_PATH
            )
            print("Database created successfully and saved to disk.")
        except Exception as e:
            return f"Error building the vector database: {e}"
    
    #Load the existing database from disk 
    try:
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant information found in the course notes for that query."
        return "\n\n".join(doc.page_content for doc in docs)
        
    except Exception as e:
        return f"An error occurred while retrieving from the notes database: {e}"
    
# Set up the Agent state and it's tools
tools = [
    effective_interest_from_nominal_interest,
    effective_discount_from_nominal_interest,
    nominal_interest_from_effective_interest,
    nominal_discount_from_effective_discount,
    effective_interest_from_nominal_discount, 
    nominal_discount_from_effective_interest,
    effective_interest_from_force_of_interest,
    force_of_interest_from_effective_interest,
    real_rate_of_interest,
    money_rate_of_interest_from_real_inflation,
    present_value_arrear_annuity,
    future_value_arrear_annuity,
    present_value_advance_annuity,
    future_value_advance_annuity,
    present_value_continuous_annuity,
    future_value_continuous_annuity,
    present_value_pthly_arrear_annuity,
    future_value_pthly_arrear_annuity,
    present_value_pthly_advance_annuity,
    future_value_pthly_advance_annuity,
    discrete_increasing_annuity_arrear,
    discrete_increasing_annuity_advance,
    discrete_increasing_annuity_continuous,
    perpuitty_arrear,
    perpuitty_advance,
    pv_perpetuity_from_d1,
    pv_perpetuity_from_d0,
    price_bond_with_tax,
    loan_prospective,
    course_notes_retriever
]

tool_executor = ToolExecutor(tools)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

model = ChatGoogleGenerativeAI(
    temperature=0, 
    model="gemini-2.5-pro", 
    google_api_key=api_key,
    top_p=0.95,
    max_output_tokens=8192)

model = model.bind_tools(tools)

# Rate limiter: 1 request per minute.
class TokenBucket:
    def __init__(self, rate_per_minute: int):
        self.rate_per_minute = max(1, int(rate_per_minute))
        self.refill_interval = 60.0 / self.rate_per_minute
        self.lock = threading.Lock()
        self.last = time.monotonic() - self.refill_interval

    def wait_for_token(self):
        with self.lock:
            now = time.monotonic()
            next_allowed = self.last + self.refill_interval
            if now < next_allowed:
                time.sleep(next_allowed - now)
            self.last = time.monotonic()

_request_bucket = TokenBucket(rate_per_minute=1)

def limited_model_invoke(messages, *args, **kwargs):
    """Rate-limited caller for the model. Do NOT assign this onto model.invoke.
    Catch API/quota errors and return an AIMessage so the graph can handle them."""
    _request_bucket.wait_for_token()
    try:
        return model.invoke(messages, *args, **kwargs)
    except Exception as e:
        # Log for debugging and return an AIMessage so the workflow does not crash.
        err_text = f"Model invocation failed: {type(e).__name__}: {e}"
        print(err_text)
        return AIMessage(content=err_text)

# NOTE: Do NOT attempt to set model.invoke or tool_executor.invoke here.

# wrap tool_executor.invoke similarly
_orig_tool_invoke = getattr(tool_executor, "invoke", None)
def limited_tool_invoke(tool_call):
    _request_bucket.wait_for_token()
    return _orig_tool_invoke(tool_call)

if _orig_tool_invoke:
    tool_executor.invoke = limited_tool_invoke

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 3. DEFINE THE NODES AND EDGES FOR THE LANGGRAPH ---

def call_actuarial_agent(state: AgentState):
    messages = state["messages"]
    response = limited_model_invoke(messages)
    return {"messages": [response]}

def handle_greeting(state: AgentState):
    last_message = state["messages"][-1].content
    response = AIMessage(content=f"Hello! I am an actuarial agent. I can help with finance math problems. You said: '{last_message}'")
    return {"messages": [response]}

def route_message(state: AgentState):
    last_message = state["messages"][-1].content.lower()
    if "hello" in last_message or "hi" in last_message:
        return "greeting"
    else:
        return "actuarial_agent"

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None) or len(last_message.tool_calls) == 0:
        return "end"
    return "continue"

def call_tool(state: AgentState):
    last_message = state["messages"][-1]
    
    tool_messages = []
    for tool_call in last_message.tool_calls:
        try:
            # wait for token before invoking the tool to respect RPM limits
            _request_bucket.wait_for_token()
            result = tool_executor.invoke(tool_call)
            action_content = str(result)
        except Exception as e:
            action_content = f"Tool execution error: {e}"
        
        # Append a ToolMessage for each tool call result
        tool_messages.append(ToolMessage(content=action_content, tool_call_id=tool_call["id"]))
        
    # Return a list of all tool results
    return {"messages": tool_messages}

def reflection_node(state: AgentState):
    last_message = state["messages"][-1]
    reflection_prompt = HumanMessage(
        content=f"""You tried to use a tool and it failed with the following error:
        {last_message.content}
        Please reflect on this error. Analyze your previous reasoning and the available tools.
        Then, create a new, corrected plan to solve the original problem. Do not apologize.
        Simply provide the corrected tool call in your response.
        """
    )
    return {"messages": [reflection_prompt]}

def decide_after_action(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and "Tool execution error:" in last_message.content:
        return "reflect"
    else:
        return "agent"

# Build the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("actuarial_agent", call_actuarial_agent)
workflow.add_node("action", call_tool)
workflow.add_node("greeting_handler", handle_greeting)
workflow.add_node("reflection_node", reflection_node)

workflow.set_conditional_entry_point(
    route_message,
    {
        "actuarial_agent": "actuarial_agent",
        "greeting": "greeting_handler",
    },
)

workflow.add_conditional_edges(
    "actuarial_agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "action",
    decide_after_action,
    {
        "agent": "actuarial_agent",
        "reflect": "reflection_node"
    }
)
workflow.add_edge("reflection_node", "actuarial_agent")
workflow.add_edge("greeting_handler", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # Generate and save the graph visualization
    # I had trouble loading langgraph, so I used mermaid code, which you can visualize online
    try:
        mermaid_syntax = app.get_graph().draw_mermaid()
        markdown_content = f"## Agent Graph Visualization\n\n```mermaid\n{mermaid_syntax}\n```"
        with open("agent_graph.md", "w") as f:
            f.write(markdown_content)
        print("Graph visualization saved to agent_graph.md")
    except Exception as e:
        print(f"Could not generate graph visualization: {e}")

    system_prompt = (
        "You are an eager actuarial specialist in the ifoa ct1/cm1 and ASSA A211 syllabus. "
        "Use your deep reasoning skills and break down the complex problem such that you can best use your tools. "
        "Always use the tools available to you to perform calculations."
        "Never perform any caclculations yourself it a tool is available for that purpose."
        "If course material are available in course_notes_retriever, consult them for each question."
        "Give concise answers, with all key reasoning steps shown, just like you would in an exam. "
        "If no information is given about the interest rate, assume it is the effective interest rate per annum."
        "Unless otherwise specified, bonds or securities are valued per 100 R/$/£ nominal value."
        "Explicitly state any assumptions you make in your reasoning."
        "Maintain decimal accuracy of at least 6 decimal places in all intermediate calculations, "
        "and the final answer should be rounded to 4 decimal places unless otherwise specified."
        "If you encounter an error when using a tool, reflect on the error and adjust your approach accordingly."
        "You may need to think outside of the box, but do not stray from the syllabus content."
        "When using the course_notes_retriever, try broad, keyword-based queries first (e.g., 'capital gain'). "
        "If that fails, try a more specific query. State the results of your search in your reasoning."
        "If no relevant resluts are found, state that in your reasoning and proceed with the information you have."
        
    )

    # Use a unique ID for this conversation to maintain memory
    conversation_config = {"configurable": {"thread_id": "actuarial-thread-live-1"}}
    
    # --- Turn 1: Initial Problem ---
    print("\n--- Running Initial Problem (Turn 1) ---")
    problem_text = (
        "Find the price of a bond that pays half-yearly coupons of 5% of nominal value per annum, and is redeemable at " \
        "par in 10 years. The effective interest rate is 6% per annum. " \
        "The investor is subject to 20% income tax and " \
        "25% capital gains tax, refer to my notes to find the appropriate test for capital gains, " \
        "to determine if capital gains is payable on redemption. "  
    )
    inputs = {"messages": [
        SystemMessage(content=system_prompt),
        HumanMessage(content=problem_text)
    ]}
    
    final_answer_1 = "Solution could not be determined."

    # guard against infinite cycles: max_turns and simple repetition detector
    max_turns = 30
    seen_agent_texts = set()
    turns = 0
    for output in app.stream(inputs, config=conversation_config):
        turns += 1
        if turns > max_turns:
            print(f"Reached max_turns ({max_turns}), stopping to avoid loop.")
            break
        stop_outer = False
        for key, value in output.items():
            print(f"Output from node '{key}':")
            if msgs := value.get("messages"):
                last_message = msgs[-1]
                # debug info
                tc = getattr(last_message, "tool_calls", None)
                print(f"  type={type(last_message).__name__} tool_calls={len(tc) if tc else 0}")
                print(last_message)
                if key == 'actuarial_agent' and not getattr(last_message, "tool_calls", None):
                    final_answer_1 = last_message.content
                    # repetition check: stop if same final agent answer repeats
                    txt_hash = hash(final_answer_1)
                    if txt_hash in seen_agent_texts:
                        print("Detected repeated final agent output — stopping stream.")
                        stop_outer = True
                        break
                    seen_agent_texts.add(txt_hash)
            else:
                print("<no messages>")
            print("---")
        if stop_outer:
            break

    # --- Turn 2: Follow-up Question ---
    print("\n--- Running Follow-up Question (Turn 2) ---")
    follow_up_text = (
        "Great. You may now assume that there is no tax payable and that the security "
        "pays coupons increasing at a contstant rate of 1% per annum to perpetuity. "
        "The first coupon payment is 5% of nominal value, and coupons are now paid annually in arrears. "
        "The annual inflation rate is expected to be 3% p.a to pepertuity. "
        "Find the price of the security evaluated at the real rate of intrest, if the money rate of interest is 6% per annum."
    )
    inputs_follow_up = {"messages": [HumanMessage(content=follow_up_text)]}

    final_answer_2 = "Follow-up solution could not be determined."

    max_turns = 30
    turns = 0
    for output in app.stream(inputs_follow_up, config=conversation_config):
        turns += 1
        if turns > max_turns:
            print(f"Reached max_turns ({max_turns}) for follow-up, stopping.")
            break
        for key, value in output.items():
            print(f"Output from node '{key}':")
            if msgs := value.get("messages"):
                last_message = msgs[-1]
                tc = getattr(last_message, "tool_calls", None)
                print(f"  type={type(last_message).__name__} tool_calls={len(tc) if tc else 0}")
                print(last_message)
                if key == 'actuarial_agent' and not getattr(last_message, "tool_calls", None):
                    final_answer_2 = last_message.content
            else:
                print("<no messages>")
            print("---")

    # --- Append the solutions to the Markdown file ---
    try:
        solution_content = (
            "\n\n## Agent's Solutions\n\n"
            f"**Problem 1:**\n`{problem_text}`\n\n"
            f"**Answer 1:**\n```\n{final_answer_1}\n```\n\n"
            f"**Problem 2 (Follow-up):**\n`{follow_up_text}`\n\n"
            f"**Answer 2:**\n```\n{final_answer_2}\n```"
        )
        with open("agent_graph.md", "a", encoding="utf-8") as f:
            f.write(solution_content)
        print("\nSolutions appended to agent_graph.md")
    except Exception as e:
        print(f"Could not append solution to markdown file: {e}")


