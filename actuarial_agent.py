import math
import os
import operator
import threading
import time
import sys
from typing import Annotated, List, Literal, TypedDict, Union
import pandas as pd
import openpyxl
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.utilities import PythonREPL
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StoreBackend, CompositeBackend
from langgraph.store.memory import InMemoryStore
import matplotlib

matplotlib.use('Agg') # Prevent GUI errors

# Load environment variables from .env file if it exists
from pathlib import Path
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

repl = PythonREPL()

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
    i = (1 + i_prime) * (1 + j) - 1 # FIX: Corrected formula
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
def pv_equity_increasing_dividends_constant_rate_perpetuity(
    D: float, g: float, i: float
) -> float:
    """
    Present value of an equity with dividends increasing at a constant rate, paid indefinitely.
    Args:
        D: The dividend payment at the end of the first period.
        g: The constant growth rate of dividends (as a decimal).
        i: The effective interest rate (as a decimal).
    """
    if i <= g:
        raise ValueError("Interest rate must be greater than growth rate for this calculation.")
    P = (D * (1 + g)) / (i - g) 
    return P

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
def loan_retrospective(
    L_0: float, i: float, n: int, X: float
) -> float:
    """
    Calculates the loan outstanding at time n using the Retrospective Method.
    Formula: L_n = L_0 * (1+i)^n - X * s_n
    
    Args:
        L_0: Initial loan amount.
        i: Effective interest rate per period.
        n: Number of periods elapsed.
        X: Periodic repayment amount (assumed level).
        
    Returns:
        The outstanding capital at time n.
    """
    accumulation_factor = (1 + i) ** n
    s_n = (accumulation_factor - 1) / i
    return L_0 * accumulation_factor - X * s_n

@tool
def calculate_irr(cashflows: list[float], times: list[float]) -> float:
    """
    Calculates the Internal Rate of Return (IRR) for a series of cashflows.
    Uses the Newton-Raphson approximation method.
    
    Args:
        cashflows: List of cashflow amounts (- for outflows, + for inflows).
        times: List of times at which cashflows occur.
        
    Returns:
        The annual effective IRR (as a decimal).
    """
    if len(cashflows) != len(times):
        raise ValueError("Cashflows and times must have the same length")
    
    # 1. Initial Guess (10%)
    rate = 0.1
    tolerance = 1e-6
    max_iterations = 50
    
    for _ in range(max_iterations):
        pv = sum(cf * (1 + rate)**(-t) for cf, t in zip(cashflows, times))
        derivative = sum(-t * cf * (1 + rate)**(-t - 1) for cf, t in zip(cashflows, times))
        
        if abs(pv) < tolerance:
            return rate
            
        rate = rate - pv / derivative
        
    return rate

@tool 
def money_cashflow_index_lagged(
    cashflows: float, index_cf_lagged: float, index_base_lagged: float
) -> float:
    """
    Adjusts cash flows for an inflation index linked security, with a lagged index.
    Args:
        cashflows: The nominal cash flow amount.
        index_cf_lagged: The lagged index for the cash flow period. (index at cf date - lag)
        index_base_lagged: The lagged base index. (index at base date - lag)
    """
    money_cashflow = cashflows * (index_cf_lagged / index_base_lagged)
    return money_cashflow

@tool 
def real_cashflow_index(
    money_cashflows: float, index_cf: float, index_base: float
) -> float:
    """
    Adjusts the money cash flows of an index linked security to real cash flows.
    Args:
        money_cashflows: The money cash flow amount.
        index_cf: The index for the cash flow period.
        index_base: The base index.
    """
    real_cashflow = money_cashflows * (index_base / index_cf)
    return real_cashflow

@tool
def volatility_of_cashflows(
    cashflows: List[float], times: List[float], i: float
) -> float:
    """
    Discrete volatility as defined: - (PV'(i)) / PV
    For PV(i) = sum_t CF_t * (1+i)^(-t)
    PV'(i) = - sum_t CF_t * t * (1+i)^(-t-1)
    => volatility = -PV'/PV = (1/(1+i)) * ( sum_t t * CF_t * v^t ) / PV
    Returns modified duration-like measure (years).
    """
    if len(cashflows) != len(times):
        raise ValueError("cashflows and times must have same length")
    v = lambda t: (1 + i) ** (-t)
    pvs = [cf * v(t) for cf, t in zip(cashflows, times)]
    pv_total = sum(pvs)
    if pv_total == 0:
        return 0.0
    weighted_sum = sum(t * cf * v(t) for cf, t in zip(cashflows, times))
    volatility = (1.0 / (1.0 + i)) * (weighted_sum / pv_total)
    return volatility

@tool
def discounted_mean_term(
    cashflows: List[float], times: List[float], i: float
) -> float:
    """
    Discounted mean term (Macaulay duration): (1/PV) * sum_t t * CF_t * v^t
    Returns weighted average time in years.
    """
    if len(cashflows) != len(times):
        raise ValueError("cashflows and times must have same length")
    v = lambda t: (1 + i) ** (-t)
    pvs = [cf * v(t) for cf, t in zip(cashflows, times)]
    pv_total = sum(pvs)
    if pv_total == 0:
        return 0.0
    mean_term = sum(t * cf * v(t) for cf, t in zip(cashflows, times)) / pv_total
    return mean_term

@tool
def discounted_convexity(
    cashflows: List[float], times: List[float], i: float
) -> float:
    """
    Discrete convexity as defined: PV''(i) / PV
    For PV''(i) = sum_t CF_t * t*(t+1) * (1+i)^(-t-2)
    => convexity = (1/(1+i)^2) * ( sum_t t*(t+1) * CF_t * v^t ) / PV
    Returns convexity in years^2.
    """
    if len(cashflows) != len(times):
        raise ValueError("cashflows and times must have same length")
    v = lambda t: (1 + i) ** (-t)
    pvs = [cf * v(t) for cf, t in zip(cashflows, times)]
    pv_total = sum(pvs)
    if pv_total == 0:
        return 0.0
    numerator = sum(t * (t + 1) * cf * v(t) for cf, t in zip(cashflows, times))
    convexity = (1.0 / ((1.0 + i) ** 2)) * (numerator / pv_total)
    return convexity

@tool
def forward_rate_from_spot_rates(
    y_t_plus_r: float, y_t: float, t: float, r: float
) -> float:
    """
    Calculates the annual effective forward rate applicable to the period from time t to time t+r.
    
    Formula: (1 + f_t_r)^r = (1 + y_{t+r})^{t+r} / (1 + y_t)^t
    
    Args:
        y_t_plus_r: Spot rate (annual effective) for term t+r.
        y_t: Spot rate (annual effective) for term t.
        t: Start time of the forward period.
        r: Duration of the forward period.
        
    Returns:
        The annual effective forward rate f_{t,r}.
    """
    return ((1 + y_t_plus_r)**(t + r) / (1 + y_t)**t)**(1/r) - 1

@tool
def spot_rate_from_forward_rate(
    y_t: float, f_t_r: float, t: float, r: float
) -> float:
    """
    Calculates the spot rate for term t+r given the spot rate for term t and the forward rate.
    
    Formula: (1 + y_{t+r})^{t+r} = (1 + y_t)^t * (1 + f_{t,r})^r
    
    Args:
        y_t: Spot rate (annual effective) for term t.
        f_t_r: Forward rate (annual effective) from t to t+r.
        t: Time t.
        r: Duration of forward period.
        
    Returns:
        The annual effective spot rate y_{t+r}.
    """
    return ((1 + y_t)**t * (1 + f_t_r)**r)**(1/(t + r)) - 1

@tool
def present_value_from_spot_rates(
    cashflows: list[float], times: list[float], spot_rates: list[float]
) -> float:
    """
    Calculates the Present Value (PV) of a series of cashflows using a list of corresponding spot rates.
    
    Formula: PV = Sum [ CF_i * (1 + y_{t_i})^{-t_i} ]
    
    Args:
        cashflows: List of cashflow amounts.
        times: List of times at which cashflows occur.
        spot_rates: List of annual effective spot rates corresponding to each time (y_{t_1}, y_{t_2}, ...).
        
    Returns:
        The total Present Value.
    """
    if not (len(cashflows) == len(times) == len(spot_rates)):
        raise ValueError("All input lists must have the same length.")
        
    pv = sum(cf * (1 + y)**(-t) for cf, t, y in zip(cashflows, times, spot_rates))
    return pv
    

@tool
def course_notes_retriever(query: str) -> str:
    """
    You can add your own notes for the agent to use. If you use a local LLM, you can use the 
    actual course material.
    """
    notes_file = "Notes.pdf"
    
    try:
        # 1. Load your course notes PDF
        loader = PyPDFLoader(notes_file)
        documents = loader.load()

        # 2. Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 3. Create embeddings and store in a Chroma vector database
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

        # 4. Create a retriever and find relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        # Format the retrieved documents into a single string
        return "\n\n".join(doc.page_content for doc in docs)

    except FileNotFoundError:
        # If the PDF is not found, return a helpful message to the agent
        return f"Error: The course notes file '{notes_file}' was not found. I cannot answer from the notes."
    except Exception as e:
        # Catch any other potential errors during the process
        return f"An unexpected error occurred while accessing the course notes: {e}"
    
@tool
def read_excel_summary(filename: str) -> str:
    """
    Reads an Excel file and returns a summary: column names, data types, 
    and the first 5 rows. Use this to understand the structure of a dataset.
    """
    try:
        df = pd.read_excel(filename)
        summary = f"**Columns:** {list(df.columns)}\n"
        summary += f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns\n"
        summary += f"**Head (First 5 rows):**\n{df.head().to_markdown(index=False)}"
        return summary
    except Exception as e:
        return f"Error reading Excel file: {e}"
    
@tool
def python_excel_interpreter(code: str) -> str:
    """
    A Python shell. Use this to execute python commands to analyze data or manipulate Excel files.
    Input should be a valid python script.
    
    ALWAYS print the final result using `print(...)` so you can see the output.
    
    Pre-installed libraries:
    - pandas (as pd): for data analysis and CSV/Excel reading.
    - openpyxl: for Excel formatting, formulas, and cell manipulation.
    - math: for standard math operations.
    - numpy (as np): for numerical operations.
    
    Example input:
    "import pandas as pd
    df = pd.read_excel('data.xlsx')
    print(df.describe())"
    """
    try:
        result = repl.run(code)
        
        if not result:
            return "Code executed successfully, but no output was printed. Did you forget to print the final answer?"
            
        return f"Output:\n{result}"
        
    except Exception as e:
        return f"Python Execution Error: {e}"
    
@tool
def python_sandbox(code: str) -> str:
    """
    A Python shell for testing new actuarial formulas or custom functions, not defined as tools.
    Use this to define a function and test it with dummy data to see if it works.

    Pre-installed libraries:
    - math: for standard math operations.
    - pandas (as pd): for data analysis 
    - numpy (as np): for numerical operations.
    """
    repl = PythonREPL()
    try:
        # Wrap in a try-except to catch syntax errors in the agent's code
        result = repl.run(code)
        return f"Execution Result:\n{result}"
    except Exception as e:
        return f"Error in code: {e}"
    
@tool
def save_custom_formula(function_code: str, description: str) -> str:
    """
    Saves a working Python function to 'custom_formulas.py'.
    Use this ONLY after you have verified the formula works in the experimental_python_lab.
    """
    filename = "custom_formulas.py"
    
    # Append the code to the file
    with open(filename, "a") as f:
        f.write("\n\n# " + description + "\n")
        f.write(function_code)
        
    return f"Success: Saved formula to {filename}. The user can now inspect it."
    
math_tools = [
    effective_interest_from_nominal_interest, effective_discount_from_nominal_interest,
    nominal_interest_from_effective_interest, nominal_discount_from_effective_discount,
    effective_interest_from_nominal_discount, nominal_discount_from_effective_interest,
    effective_interest_from_force_of_interest, force_of_interest_from_effective_interest,
    real_rate_of_interest, money_rate_of_interest_from_real_inflation,
    present_value_arrear_annuity, future_value_arrear_annuity,
    present_value_advance_annuity, future_value_advance_annuity,
    present_value_continuous_annuity, future_value_continuous_annuity,
    present_value_pthly_arrear_annuity, future_value_pthly_arrear_annuity,
    present_value_pthly_advance_annuity, future_value_pthly_advance_annuity,
    discrete_increasing_annuity_arrear, discrete_increasing_annuity_advance,
    discrete_increasing_annuity_continuous, perpuitty_arrear, perpuitty_advance,
    pv_equity_increasing_dividends_constant_rate_perpetuity, price_bond_with_tax,
    loan_prospective, loan_retrospective, volatility_of_cashflows, discounted_mean_term, discounted_convexity,
    forward_rate_from_spot_rates, spot_rate_from_forward_rate, present_value_from_spot_rates, calculate_irr,
    course_notes_retriever
]

lab_tools = [
    course_notes_retriever, python_excel_interpreter, 
    read_excel_summary, python_sandbox, save_custom_formula
]

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")


# Supervisor Model (Gemini 3-based) - Used for routing decisions, In accordance with the google documentation
supervisor_model = ChatGoogleGenerativeAI(
    temperature=1.0,  
    model="gemini-3-pro-preview",  
    thinking_level="high",  
    include_thoughts=True,  
    google_api_key=api_key,
    top_p=0.95,  
    max_output_tokens=18000  
)

# Worker Model (Gemini 2.5-based) - Used for Math and Lab agents
worker_model = ChatGoogleGenerativeAI(
    temperature=0,  
    model="gemini-2.5-flash-preview-09-2025",  
    google_api_key=api_key,
    top_p=0.95,  
    max_output_tokens=12000  
)

math_agent = create_agent(
    model=worker_model,  # Using Gemini 2.5 for worker tasks 
    tools=math_tools,
    system_prompt="You are an eager actuarial specialist in the IFOA CT1/CM1 and ASSA A211 syllabus. "
                    "Your goal is to solve financial mathematics problems with high precision.\n"
                    "STRICT FORMATTING & BEHAVIOR RULES:\n"
                    "1. **Exam Style**: Give concise answers with all key reasoning steps shown, just like in an exam.\n"
                    "The final answer must be rounded to 4 decimal places unless otherwise specified.\n"
                    "3. **Assumptions**: If no information is given about the interest rate, assume it is the effective interest rate per annum. "
                    "Explicitly state any assumptions you make.\n"
                    "4. **Bonds**: Unless otherwise specified, bonds are valued per 100 R/$/£ nominal value.\n"
                    "5. **Tool Usage**: Do not attempt to find numeric solutions manually if you can use a tool. \n"
                    "formatting.\n"
                    "\n"
                    "STRICT FORMATTING RULES:\n"
                    "1. **LaTeX**: Use LaTeX for ALL mathematical variables and expressions. "
                    "   - Inline: $x$, $i^{(12)}$, $\ddot{a}_{n|}$.\n"
                    "   - Block: $$ PV = \dots $$. Ensure $$ are on their own lines.\n"
                    "2. **Structure**: Organize solutions with clear Markdown headers:\n"
                    "   - `## Step 1: Definition of Variables`\n"
                    "   - `## Step 2: Equation of Value`\n"
                    "   - `## Step 3: Calculation`\n"
                    "3. **Tables**: Use Markdown tables for cashflow schedules or time diagrams.\n"
                    "4. **Precision**: Keep 6+ decimal places for intermediate steps, round final answer to 2 or 4 (currency vs rate).\n"
                    "5. **Explanations**: Briefly explain *why* a formula is used (e.g. 'Since payments are in advance...').\n"
                    "6. **Final Answer**: End with a clear box: `> **Final Answer:** `.\n"
)

# Setup Composite Backend
composite_backend = lambda rt: CompositeBackend(
    default=FilesystemBackend(root_dir="."), # Real file access to current dir
    routes={
        "/memories/": StoreBackend(rt),      # Persistent memory access
    }
)


lab_agent = create_deep_agent(
    model=worker_model,  # Using Gemini 2.5 for worker tasks
    tools=lab_tools,
    backend=composite_backend,
    store=InMemoryStore(),
    system_prompt="You are a Python Data Lab Specialist. " \
    "Your role is to AUTOMATICALLY execute Python code to generate Excel files, charts, and analyze data.\n" \
    "STRICT RULES:\n"  \
    "1. When you receive Python code from another agent, EXECUTE IT IMMEDIATELY using python_sandbox.\n" \
    "2. If code generation is requested, write AND execute it - don't ask for permission.\n" \
    "3. Use matplotlib for visualizations and save all outputs to files.\n" \
    "4. Report results after execution.\n" \
   
)

class SupervisorState(MessagesState):
    """State containing the conversation history and the routing decision."""
    next: str

system_prompt_supervisor = (
    "You are the Supervisor of an actuarial team.\n"
    "Your goal is to delegate tasks to the most appropriate specialist.\n"
    "\n"
    "Before routing, briefly (1-2 lines) STATE YOUR PLAN or TO-DO LIST to the user.\n"
    "Then, select the next worker:\n"
    "1. **Math_Specialist**: For calculations involving annuities, bonds, loans, rates, or project appraisal.\n"
    "2. **Lab_Specialist**: For generating Python scripts, Excel files, charts, or checking course notes.\n"
    "3. **FINISH**: When the answer is complete and verified.\n"
    "\n"
    "Respond with the name of the next worker (Math_Specialist, Lab_Specialist, or FINISH)."
)

def supervisor_node(state: SupervisorState):
    messages = state["messages"]
    response = supervisor_model.invoke(  # Using Gemini 3 for routing decisions
        [SystemMessage(content=system_prompt_supervisor)] + messages
    )
    
    # Gemini 3 returns content as a list, Gemini 2.5 returns as string
    if isinstance(response.content, list):
        # Extract text from the list of content parts
        full_response = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
    else:
        full_response = response.content.strip()
    
    # Robust parsing: Look for keywords in the response
    if "Math_Specialist" in full_response:
        next_step = "Math_Specialist"
    elif "Lab_Specialist" in full_response:
        next_step = "Lab_Specialist"
    elif "FINISH" in full_response:
        next_step = "FINISH"
    else:
        next_step = "FINISH"  # Default fallback
        
    return {"next": next_step}

def call_math_agent(state: SupervisorState):
    # Invoke the math specialist with retry logic for MALFORMED_FUNCTION_CALL errors
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\n=== RETRY {attempt}/{max_retries} for Math Agent ===")
                time.sleep(retry_delay * attempt)  # Exponential backoff
            
            print(f"\n=== DEBUG: Math Agent Input State ===")
            print(f"State keys: {state.keys()}")
            print(f"Messages count: {len(state.get('messages', []))}")
            
            response = math_agent.invoke(state)
            
            print(f"\n=== DEBUG: Math Agent Response ===")
            print(f"Response type: {type(response)}")
            print(f"Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
            print(f"Messages in response: {len(response.get('messages', []))}")
            
            if "messages" in response and response["messages"]:
                last_msg = response["messages"][-1]
                print(f"Last message type: {type(last_msg)}")
                print(f"Last message has content attr: {hasattr(last_msg, 'content')}")
                
                # Check for MALFORMED_FUNCTION_CALL
                if hasattr(last_msg, 'response_metadata'):
                    finish_reason = last_msg.response_metadata.get('finish_reason', '')
                    print(f"Finish reason: {finish_reason}")
                    
                    if finish_reason == 'MALFORMED_FUNCTION_CALL' and attempt < max_retries - 1:
                        print(f"Encountered MALFORMED_FUNCTION_CALL, retrying...")
                        continue  # Retry
                
                print(f"Last message content: {last_msg.content if hasattr(last_msg, 'content') else 'NO CONTENT ATTR'}")
                print(f"Last message repr: {repr(last_msg)[:200]}")
            
            # Return only the last message (the result) to append to the main history
            return {"messages": [response["messages"][-1]]}
            
        except Exception as e:
            print(f"ERROR in Math Agent: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying due to exception...")
                continue
            return {"messages": [AIMessage(content=f"Error in math agent after {max_retries} attempts: {str(e)}")]}
    
    # If all retries failed with MALFORMED_FUNCTION_CALL
    return {"messages": [AIMessage(content="Math agent failed after multiple retries due to MALFORMED_FUNCTION_CALL. Please try rephrasing your question.")]}

def call_lab_agent(state: SupervisorState):
    # Invoke the lab specialist with retry logic for MALFORMED_FUNCTION_CALL errors
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\n=== RETRY {attempt}/{max_retries} for Lab Agent ===")
                time.sleep(retry_delay * attempt)  # Exponential backoff
            
            print(f"\n=== DEBUG: Lab Agent Input State ===")
            print(f"State keys: {state.keys()}")
            print(f"Messages count: {len(state.get('messages', []))}")
            
            response = lab_agent.invoke(state)
            
            print(f"\n=== DEBUG: Lab Agent Response ===")
            print(f"Response type: {type(response)}")
            print(f"Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
            print(f"Messages in response: {len(response.get('messages', []))}")
            
            if "messages" in response and response["messages"]:
                last_msg = response["messages"][-1]
                print(f"Last message type: {type(last_msg)}")
                print(f"Last message has content attr: {hasattr(last_msg, 'content')}")
                
                # Check for MALFORMED_FUNCTION_CALL
                if hasattr(last_msg, 'response_metadata'):
                    finish_reason = last_msg.response_metadata.get('finish_reason', '')
                    print(f"Finish reason: {finish_reason}")
                    
                    if finish_reason == 'MALFORMED_FUNCTION_CALL' and attempt < max_retries - 1:
                        print(f"Encountered MALFORMED_FUNCTION_CALL, retrying...")
                        continue  # Retry
                
                print(f"Last message content length: {len(last_msg.content) if hasattr(last_msg, 'content') else 'NO CONTENT ATTR'}")
                print(f"Last message content preview: {last_msg.content[:500] if hasattr(last_msg, 'content') else 'NO CONTENT ATTR'}")
            
            return {"messages": [response["messages"][-1]]}
            
        except Exception as e:
            print(f"ERROR in Lab Agent: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying due to exception...")
                continue
            return {"messages": [AIMessage(content=f"Error in lab agent after {max_retries} attempts: {str(e)}")]}
    
    # If all retries failed with MALFORMED_FUNCTION_CALL
    return {"messages": [AIMessage(content="Lab agent failed after multiple retries due to MALFORMED_FUNCTION_CALL. Please try rephrasing your question.")]} 

builder = StateGraph(SupervisorState)

# Add Nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("Math_Specialist", call_math_agent)
builder.add_node("Lab_Specialist", call_lab_agent)

# Add Edges
builder.add_edge(START, "supervisor")

# Conditional Edge from Supervisor to Specialists
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Math_Specialist": "Math_Specialist",
        "Lab_Specialist": "Lab_Specialist",
        "FINISH": END
    }
)

# Specialists report back to supervisor to decide next steps (The Loop)
builder.add_edge("Math_Specialist", "Lab_Specialist")
builder.add_edge("Lab_Specialist", END)

# Compile Graph with Memory
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

class DualLogger:
    """Redirects stdout to both terminal and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    
    # 1. Setup Dual Logging to capture Exam-Style answers
    sys.stdout = DualLogger("agent_output.md")
    
    # Use a unique thread ID for conversation persistence
    config = {"configurable": {"thread_id": "actuarial-supervisor-live-1"}, "recursion_limit": 10}
    
    print("\n--- ACTUARIAL AGENT (SUPERVISOR MODE) ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        print("\nEnter your actuarial question (or press Enter for default test):")
        try:
            user_input = input("> ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input.strip():
                query = user_input
            else:
                print("Using default test calculation...")
                query = "Calculate the first quarterly repayment for a loan of R1,000,000 over 15 years " \
                "that increases every year by 5%, using a nominal interest rate of 10% compounded monthly. " \
                "Then generate an Excel file named loan_schedule.xlsx with the full amortization schedule " \
                "(Opening Balance, Interest, Repayment, Capital, Closing Balance) and create a matplotlib chart."
            
            print(f"\nUser: {query}")
            
            # Update inputs with new query
            inputs = {"messages": [("user", query)]}
            
            # Run the graph
            for update in app.stream(inputs, config=config):
                for node, values in update.items():
                    print(f"--- Node: {node} ---")
                    if "messages" in values:
                        print(values["messages"][-1].content)
                        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
            
    print("Agent execution complete")