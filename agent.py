import asyncio
import asyncpg
import uuid
import os
import itertools
import json 
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import httpx
from typing import TypedDict, Annotated, Literal, cast, Optional, List
from contextlib import asynccontextmanager

from placement import placement_Retriever
from qdrant import RulesRetriever
from orcr import ORCR_Retriever
from user_crud_asyncpg import (
    init_db_pool, close_db_pool, get_pool,
    get_by_email, create_user, update_user,
    create_schema, usage_schema, upadate_schema,
    Category, Gender, get_by_id,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send                     
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from collections import defaultdict

GROQ_KEYS_RAW = os.getenv("GROQ_API_KEY", "")
SERPER_KEY    = os.getenv("SERPER_API_KEY", "")
mmodel      = "openai/gpt-oss-120b" 
SUMMARISER_MODEL = "llama-3.3-70b-versatile"


class APIKeyRotator:
    def __init__(self, raw: str):
        self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("No API keys provided.")
        self._cycle = itertools.cycle(self.keys)
        print(f"[KeyRotator] Loaded {len(self.keys)} key(s).")

    def next(self) -> str:
        return next(self._cycle)
    
_guest_sessions: dict[str, dict] = defaultdict(lambda: {"short_term_memory": [], "summary": ""})    

groq_rotator = APIKeyRotator(GROQ_KEYS_RAW)
db_retriever        = ORCR_Retriever()       
rules_retriever     = RulesRetriever()       
placement_retriever = placement_Retriever()  

@tool
async def retrieve_college_allocations_JEE_Adv(rank: int, category: str, gender: str) -> str:
    """
    Fetches IIT seat allocations from the JoSAA database for a given
    JEE Advanced rank, category, and gender.
    Use this for ANY JEE-Advanced / IIT college prediction query.
    Do NOT answer from your own knowledge - only use data returned here.
    Parameters
    rank     : Candidate's JEE Advanced rank (integer).
    category : One of OPEN, OBC-NCL, GEN-EWS, SC, ST.
    gender   : "Gender-Neutral" or "Female Only".
    """
    print(f"[TOOL] JEE_Adv  rank={rank}, cat={category}, gen={gender}")
    try:
        rows = await db_retriever.runa(rank, category, gender)
        if not rows:
            return "No IIT allocations found for the given criteria."
        return "JEE Advanced DB allocations:\n" + str(rows)
    except Exception as e:
        return f"JEE Advanced DB lookup failed: {e}"


@tool
async def retrieve_college_allocations_JEE_Main(rank: int, category: str, gender: str) -> str:
    """
    Fetches NIT/IIIT seat allocations from the JoSAA database for a given
    JEE Main rank, category, and gender.
    Use this for ANY JEE-Main / NIT / IIIT college prediction query.
    Do NOT answer from your own knowledge.
    Parameters
    rank     : Candidate's JEE Main CRL rank (integer).
    category : One of OPEN, OBC-NCL, GEN-EWS, SC, ST.
    gender   : "Gender-Neutral" or "Female Only".
    """
    print(f"[TOOL] JEE_Main rank={rank}, cat={category}, gen={gender}")
    try:
        rows = await db_retriever.runm(rank, category, gender)
        if not rows:
            return "No NIT/IIIT allocations found for the given criteria."
        return "JEE Main DB allocations:\n" + str(rows)
    except Exception as e:
        return f"JEE Main DB lookup failed: {e}"


@tool
async def placement_data(institute: str) -> str:
    """
    Returns the latest placement statistics (median salary, top recruiters,
    placement percentage, etc.) for the specified institute.
    Parameters
    institute : Full institute name e.g. "IIT Bombay", "NIT Trichy".
    """
    print(f"[TOOL] placement_data  institute={institute}")
    try:
        rows = await placement_retriever.run(institute)
        if not rows:
            return f"No placement data found for '{institute}'."
        return f"Placement data for {institute}:\n" + "\n".join(str(r) for r in rows)
    except Exception as e:
        return f"Placement retrieval failed: {e}"


@tool
async def search_jossa(query: str) -> str:
    """
    Searches the local Qdrant index that contains official JoSAA PDFs
    (rules, seat matrix, counselling procedures, cut-offs).
    Use this for rule/process/eligibility questions.
    Parameters
    query : Natural-language question about JoSAA rules or procedures.
    """
    print(f"[TOOL] search_jossa  query={query!r}")
    try:
        chunks = await rules_retriever.search(query, 3)
        if not chunks:
            return "No relevant JoSAA rule documents found."
        return f"JoSAA rule excerpts for '{query}':\n" + "\n---\n".join(chunks)
    except Exception as e:
        return f"Qdrant search failed: {e}"


@tool
async def search_google_images(query: str) -> str:
    """
    Searches Google Images via Serper.dev and returns the top image URL.
    Use this when the user explicitly asks for a photo or image.
    Parameters
    query : Image search query string.
    """
    print(f"[TOOL] search_google_images  query={query!r}")
    url     = "https://google.serper.dev/images"
    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json={"q": query, "num": 3}, timeout=10)
            r.raise_for_status()
            images = r.json().get("images", [])
            if not images:
                return f"No images found for '{query}'."
            fi = images[0]
            return f"Image for '{query}':\nTitle: {fi.get('title')}\nURL: {fi.get('imageUrl')}"
        except Exception as e:
            return f"Image search failed: {e}"


@tool
async def search_web_serper(query: str) -> str:
    """
    Performs a live Google web search via Serper.dev and returns the top
    3 organic result snippets.  Use this for real-time / recent information.

    Parameters
    ----------
    query : Web search query string.
    """
    print(f"[TOOL] search_web_serper  query={query!r}")
    url     = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json={"q": query}, timeout=10)
            r.raise_for_status()
            organic = r.json().get("organic", [])[:3]
            snippets = [f"{i['title']}\n{i['snippet']}" for i in organic]
            if not snippets:
                return f"No web results found for '{query}'."
            return "Web search results:\n" + "\n---\n".join(snippets)
        except Exception as e:
            return f"Web search failed: {e}"


alltools: list = [
    search_google_images,
    search_web_serper,
    search_jossa,
    retrieve_college_allocations_JEE_Main,
    retrieve_college_allocations_JEE_Adv,
    placement_data,
]
tool_names: dict = {t.name: t for t in alltools}

class AgentState(TypedDict):
    messages:          Annotated[list, add_messages]
    short_term_memory: list   
    session_id:        str
    user_id:           int
    current_input:     str    
    summary:           str     
    final_response:    str 
    session_key:Optional[str]   

class OrchestratorAgent:
    def __init__(self, window_size: int = 5):
    
        self.window_size   = window_size
        self.orc_llm_cycle = None   
        self.ans_llm_cycle = None  
        self.summariser    = None
        self.graph         = None

    def initialize(self):
     
        keys = [k.strip() for k in GROQ_KEYS_RAW.split(",") if k.strip()]
        orc_instances = [
            ChatGroq(model=mmodel, temperature=0, groq_api_key=k).bind_tools(alltools)
            for k in keys
        ]
        self.orc_llm_cycle = itertools.cycle(orc_instances)
        ans_instances = [
            ChatGroq(model=mmodel, temperature=0.3, groq_api_key=k)
            for k in keys
        ]
        self.ans_llm_cycle = itertools.cycle(ans_instances)

        try:
            self.summariser = ChatGroq(model=SUMMARISER_MODEL, temperature=0, groq_api_key=keys[0])
        except Exception:
            self.summariser = ChatGroq(model=mmodel, temperature=0, groq_api_key=keys[0])

        self.graph = self._build_graph()
        print("[OrchestratorAgent] Graph compiled successfully.")

    def _build_graph(self) -> StateGraph:
   
        b = StateGraph(AgentState)
        b.add_node("search_google_images_agent", self._make_specialist_node("search_google_images"))
        b.add_node("load_memory",self._load_memory_node)
        b.add_node("orchestrator_agent",self._orchestrator_node)
        b.add_node("search_web_serper_agent",self._make_specialist_node("search_web_serper"))

        b.add_node("search_jossa_agent",self._make_specialist_node("search_jossa"))
        b.add_node("placement_agent",self._make_specialist_node("placement_data"))

        b.add_node("answering_agent",self._answering_agent_node)
        b.add_node("save_memory",self._save_memory_node)
        b.add_node("jee_main_agent",self._make_specialist_node("retrieve_college_allocations_JEE_Main"))
        b.add_node("jee_adv_agent",self._make_specialist_node("retrieve_college_allocations_JEE_Adv"))
        
        b.add_edge(START, "load_memory")
        b.add_edge("load_memory", "orchestrator_agent")
        b.add_conditional_edges(
            "orchestrator_agent",
            self._route_after_orchestrator,
            ["search_google_images_agent","search_web_serper_agent", "search_jossa_agent","jee_main_agent","jee_adv_agent","placement_agent","answering_agent",],)
        for node in [
             "search_google_images_agent","search_web_serper_agent","search_jossa_agent","search_jossa_agent","jee_main_agent","jee_adv_agent","placement_agent",]:b.add_edge(node, "orchestrator_agent")
        b.add_edge("answering_agent", "save_memory")
        b.add_edge("save_memory", END)
        return b.compile()

    async def _load_memory_node(self, state: AgentState) -> dict:
        user_profile = None
        pool = get_pool()
        if state["user_id"]!=-1 and pool:
            async with pool.acquire() as conn:
                user_profile = await get_by_id(cast(asyncpg.Connection, conn), state["user_id"])

        if user_profile:
            user_info = (
                f"- Name: {user_profile.name}\n"
                f"- JEE Advanced Rank: {user_profile.adv_rank}\n"
                f"- JEE Mains Rank: {user_profile.mains_rank}\n"
                f"- Category: {user_profile.category.value}\n"
                f"- Gender: {user_profile.gender.value}\n"
                f"- Preferred Branches: "
                f"{', '.join(user_profile.preferred_branches) if user_profile.preferred_branches else 'None'}"
            )
            rank_rules = """
            STRICT RULES:
            1. For JEE Advanced college predictions → ALWAYS call `retrieve_college_allocations_JEE_Adv` using the Advanced Rank, Category and Gender from the USER PROFILE above. Never ask the user for their rank again.
            2. For JEE Main college predictions → ALWAYS call `retrieve_college_allocations_JEE_Main` using Mains Rank, Category and Gender from the USER PROFILE.
            3. Never answer college prediction queries from your own knowledge. The database is the only authoritative source."""

        else:
            user_info = "Guest user — no profile registered."
            rank_rules = """
            STRICT RULES:
            1. The user has NOT registered. Extract rank, category, and gender DIRECTLY from their message.
            2. For JEE Advanced / IIT queries → call `retrieve_college_allocations_JEE_Adv` with rank/category/gender parsed from the conversation. NEVER ask them to register.
            3. For JEE Main / NIT / IIIT queries → call `retrieve_college_allocations_JEE_Main` with rank/category/gender parsed from the conversation.
            4. If rank or category is missing from the message, ask ONLY for the missing piece. Do not ask them to register.
            5. Never answer college prediction queries from your own knowledge. The database is the only authoritative source."""

        system_prompt = f"""You are an expert JoSAA counselling assistant. You are advised to include images in your response as users like visuals. Use search_google_images for searching images related to the topic.

                        USER PROFILE:
                        {user_info}

                        {rank_rules}
                        4. For placement stats → call `placement_data`. Never answer from web search, database is the only authoritative source.
                        5. For JoSAA rules / procedures → call `search_jossa`. Authoritative source is the database but you can search the web using 'search_web_serper' in extreme cases.
                        6. For current news / dates → call `search_web_serper`.
                        7. For images → call `search_google_images`.

                        When you have all the information you need, produce a thorough reasoning summary (no need for markdown formatting - the answering agent will handle that)."""
        
        msgs: list = [SystemMessage(content=system_prompt)]
        raw_memory = state.get("short_term_memory", [])

        for turn in raw_memory[-10:]:
            if isinstance(turn, str):
                try:
                    turn = json.loads(turn)
                except Exception:
                    continue
            if not isinstance(turn, dict):
                continue
            if turn.get("role") == "user":
                msgs.append(HumanMessage(content=turn["content"]))
            elif turn.get("role") == "ai":
                msgs.append(AIMessage(content=turn["content"]))

        msgs.append(HumanMessage(content=state["current_input"]))  

        return {"messages": msgs}

    
    async def _orchestrator_node(self, state: AgentState) -> dict:
        llm = next(self.orc_llm_cycle)
        try:
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            print(f"[orchestrator_agent] LLM error: {e}")
            return {"messages": [AIMessage(content=f" Orchestrator error: {e}")]}

    def _make_specialist_node(self, tool_name: str):
        async def specialist_node(state: AgentState) -> dict:
            print(f"[specialist] Running node for tool: {tool_name}")
            last_msg = state["messages"][-1]
            tool_func = tool_names.get(tool_name)
            results: list = []

            if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                return {"messages": results}

            for tc in last_msg.tool_calls:
                if tc["name"] != tool_name:
                    continue
                try:
                    output = await tool_func.ainvoke(tc["args"])
                except Exception as e:
                    output = f"Tool {tool_name} failed: {e}"
                results.append(
                    ToolMessage(
                        content=str(output),
                        tool_call_id=tc["id"],
                        name=tool_name,
                    )
                )
            return {"messages": results}

        specialist_node.__name__ = f"{tool_name}_node"
        return specialist_node


    def _route_after_orchestrator(self, state: AgentState):
    
        last_msg = state["messages"][-1]

        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            return "answering_agent"
        tool_to_node = {
            "search_google_images":"search_google_images_agent",
            "search_web_serper":"search_web_serper_agent",
            "search_jossa":"search_jossa_agent",
            "retrieve_college_allocations_JEE_Main":"jee_main_agent",
            "retrieve_college_allocations_JEE_Adv":"jee_adv_agent",
            "placement_data":"placement_agent",
        }

        seen_nodes = set()
        sends = []
        for tc in last_msg.tool_calls:
            node_name = tool_to_node.get(tc["name"])
            if node_name and node_name not in seen_nodes:
                seen_nodes.add(node_name)
                sends.append(Send(node_name, state))

        return sends if sends else "answering_agent"


    async def _answering_agent_node(self, state: AgentState) -> dict:
        orc_reasoning = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                orc_reasoning = msg.content
                break

        formatting_prompt = f"""You are a friendly, expert JoSAA counselling assistant.
Below is the raw reasoning from the research agent.  Your job is to rewrite it
as a clear, well-structured, markdown-formatted answer for the student.
Guidelines:
1.  Use markdown tables for college lists (columns: Institute | Branch | Opening Rank | Closing Rank).
2. Use bullet points for lists of rules or steps.
3. Use **bold** for important numbers or names.
4. Keep the tone warm and encouraging.
5. End with a short actionable tip if relevant.
RAW REASONING:
{orc_reasoning}
"""

        llm = next(self.ans_llm_cycle)
        try:
            response = await llm.ainvoke([
                    SystemMessage(content="You are a student-friendly formatter."),
                    HumanMessage(content=formatting_prompt),
                ])
            return {"messages": [response]}
        except Exception as e:
            print(f"[answering_agent] error: {e}")
            return {"messages": [AIMessage(content=orc_reasoning)]} 

    async def _save_memory_node(self, state: AgentState) -> dict:

        print("[save_memory] Persisting memory and usage.")

        final_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        raw = state.get("short_term_memory", [])
        memory = []
        for m in raw:
            if isinstance(m, str):
                try:
                    memory.append(json.loads(m))  
                except Exception:
                    pass
            elif isinstance(m, dict):
                memory.append(m)                  

        memory.append({"role": "user", "content": state["current_input"]})
        memory.append({"role": "ai",   "content": final_response})

        max_turns = self.window_size * 2
        old_summary = state.get("summary", "")
        new_summary = old_summary

        if len(memory) > max_turns * 2:
            half = len(memory) // 2
            to_summarise = memory[:half]
            memory = memory[half:]
            new_summary = await self._summarise(to_summarise, old_summary)

        pool = get_pool()
        if pool and state["user_id"]!=-1:
            async with pool.acquire() as conn:
                user = await get_by_id(cast(asyncpg.Connection, conn), state["user_id"])
                if user:
                    new_usage = usage_schema(
                        queries_today=user.usage.queries_today + 1,
                        cooldown_until=user.usage.cooldown_until,
                        last_query=datetime.now(),
                    )
                    await update_user(
                        cast(asyncpg.Connection, conn),
                        user.email,
                        upadate_schema(
                            short_term_memory=[json.dumps(m) for m in memory],
                            summary=new_summary,
                            usage=new_usage,
                        ),)
        if state["user_id"] == -1:
            session_key = state.get("session_key")
            if session_key:
                _guest_sessions[session_key]["short_term_memory"] = memory
                _guest_sessions[session_key]["summary"] = new_summary
                print(f"[save_memory] Updated guest session {session_key}")            
                            
        return {
            "short_term_memory": memory,       
            "summary": new_summary,
            "final_response": final_response,}
    async def _summarise(self, messages: list[dict], existing: str) -> str:
        if not messages:
            return existing
        conv = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        )
        prompt = (
            f"Existing summary:\n{existing or 'None'}\n\n"
            f"New conversation to incorporate:\n{conv}\n\n"
            "Write a concise summary (≤150 words) merging old and new."
        )
        try:
            r = await self.summariser.ainvoke([
                SystemMessage(content="You summarise conversations concisely."),
                HumanMessage(content=prompt),
            ])
            content = r.content
            return (content if isinstance(content, str) else str(content)).strip()
        except Exception as e:
            print(f"[summarise] Failed: {e}")
            return existing
    def get_stream(self, initial_state: dict):
        """Returns an async generator that yields (message, metadata) tuples."""
        return self.graph.astream(initial_state, stream_mode="messages")
_agent: Optional[OrchestratorAgent] = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB pool + compile LangGraph. Shutdown: close pool."""
    print("[lifespan] Starting up...")
    await init_db_pool()
    global _agent
    _agent = OrchestratorAgent(window_size=5)
    _agent.initialize()
    print("[lifespan] Ready.")
    yield
    print("[lifespan] Shutting down...")
    await close_db_pool()
app = FastAPI(lifespan=lifespan, title="JoSAA AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    query:      str
    email: Optional[str]=None
    session_id: Optional[str] = "None"
    name:Optional[str]  = None
    adv_rank:Optional[int]  = None
    mains_rank: Optional[int]  = None
    category:Optional[str]  = "OPEN"
    gender:Optional[str]  = "Male"

class CheckUserRequest(BaseModel):
    email: str

@app.post("/check-user")
async def check_user(req: CheckUserRequest):
    pool = get_pool()
    if pool is None:
        raise HTTPException(503, "Database not ready")
    async with pool.acquire() as conn:
        user = await get_by_email(conn, req.email)
    return {"exists": user is not None}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not initialised")
    
    if not req.email:
        session_key = req.session_id or str(uuid.uuid4())
        guest_mem   = _guest_sessions[session_key]

        initial_state = {
            "messages":          [],
            "short_term_memory": guest_mem["short_term_memory"],
            "session_id":        session_key,
            "user_id":           -1,
            "current_input":     req.query,
            "summary":           guest_mem["summary"],
            "final_response":    "",
            "session_key": session_key,
        }

        async def guest_streamer():
            async for message, metadata in _agent.get_stream(initial_state):
                node = metadata.get("langgraph_node", "")

                if node == "orchestrator_agent" and isinstance(message, AIMessage):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tc in message.tool_calls:
                            yield f"||TOOL_CALL:{tc['name']}||\n"

                elif node == "answering_agent" and isinstance(message, AIMessage):
                    content = message.content
                    if isinstance(content, list):
                        content = " ".join(str(p) for p in content)
                    if content:
                        yield content

                elif node == "save_memory" and isinstance(message, dict):
                    # persist in-memory for this session
                    _guest_sessions[session_key]["short_term_memory"] = message.get(
                        "short_term_memory", guest_mem["short_term_memory"]
                    )
                    _guest_sessions[session_key]["summary"] = message.get(
                        "summary", guest_mem["summary"]
                    )

        return StreamingResponse(guest_streamer(), media_type="text/plain")

    pool = get_pool()
    if pool is None:
        raise HTTPException(503, "Database not ready")

    async with pool.acquire() as conn:
        user = await get_by_email(conn, req.email)
        if user is None:
            try:
                cat_enum = Category(req.category)
            except ValueError:
                cat_enum = Category.OPEN
            try:
                gen_enum = Gender(req.gender)
            except ValueError:
                gen_enum = Gender.male

            user_data = create_schema(
                name=req.name or req.email.split("@")[0],
                email=req.email,
                adv_rank=req.adv_rank or 0,
                mains_rank=req.mains_rank,
                category=cat_enum,
                gender=gen_enum,
                preferred_branches=[],
                usage=usage_schema(queries_today=0),
                short_term_memory=[],
                summary="",
            )
            user = await create_user(conn, user_data)
            print(f"[chat] Created user: {user.name}")
        else:
            print(f"[chat] Existing user: {user.name}")

    initial_state = {
        "messages":          [],
        "short_term_memory": user.short_term_memory or [],
        "session_id":        req.session_id,
        "user_id":           user.id,
        "current_input":     req.query,
        "summary":           user.summary or "",
        "final_response":    "",
    }

    async def token_streamer():
        async for message, metadata in _agent.get_stream(initial_state):
            node = metadata.get("langgraph_node", "")

            if node == "orchestrator_agent" and isinstance(message, AIMessage):
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        yield f"||TOOL_CALL:{tc['name']}||\n"

            elif node == "answering_agent" and isinstance(message, AIMessage):
                content = message.content
                if isinstance(content, list):
                    content = " ".join(str(p) for p in content)
                if content:
                    yield content

    return StreamingResponse(token_streamer(), media_type="text/plain")
